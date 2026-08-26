// sofuu-ffi — libcurl bindings (PLAN-RUST-MIGRATION M4: HTTP client).
//
// The fetch() port (rt/http_client.rs) drives the system libcurl the same
// way the retired src/http/client.c did: one CURLM multi handle bridged to
// the libuv loop via CURLMOPT_SOCKETFUNCTION/CURLMOPT_TIMERFUNCTION.
// Option/constant values are the stable public curl ABI (curl.h, 8.x).
// curl_easy_setopt/getinfo/multi_setopt are variadic — the extern decls
// below are callable with the exact C vararg types (c_long for the LONG
// options, pointers for STRING/OBJECT/CB ones).

use std::ffi::c_char;
use std::os::raw::{c_int, c_long, c_void};

pub const CURLE_OK: c_int = 0;
pub const CURLMSG_DONE: c_int = 1;
pub const CURLM_OK: c_int = 0;

pub const CURL_GLOBAL_ALL: c_long = 3;

pub const CURL_CSELECT_IN: c_int = 0x01;
pub const CURL_CSELECT_OUT: c_int = 0x02;

pub const CURL_POLL_NONE: c_int = 0;
pub const CURL_POLL_IN: c_int = 1;
pub const CURL_POLL_OUT: c_int = 2;
pub const CURL_POLL_INOUT: c_int = 3;
pub const CURL_POLL_REMOVE: c_int = 4; // verified multi.h:287-291

/// curl_socket_t {-1} — pass to socket_action to run the internal timeout.
pub const CURL_SOCKET_TIMEOUT: c_int = -1;

// ── option ids (curl.h CURLoption / CURLINFO) ────────────────────────
pub const CURLOPT_WRITEDATA: c_int = 10001;
pub const CURLOPT_URL: c_int = 10002;
pub const CURLOPT_TIMEOUT: c_int = 13;
pub const CURLOPT_HTTPHEADER: c_int = 10023;
pub const CURLOPT_HEADERDATA: c_int = 10029;
pub const CURLOPT_CUSTOMREQUEST: c_int = 10036;
pub const CURLOPT_FOLLOWLOCATION: c_long = 52;
pub const CURLOPT_WRITEFUNCTION: c_int = 20011;
pub const CURLOPT_HEADERFUNCTION: c_int = 20079;
pub const CURLOPT_PRIVATE: c_int = 10103;
pub const CURLOPT_COPYPOSTFIELDS: c_int = 10165;
pub const CURLOPT_USERAGENT: c_int = 10018;
pub const CURLOPT_FAILONERROR: c_long = 45;
// M8 (AI module): verified curl.h — POSTFIELDS=OBJECTPOINT(10000)+15,
// POSTFIELDSIZE=LONG(0)+60, SSL_VERIFYPEER=64, MAXFILESIZE=114,
// TIMEOUT_MS=155.
pub const CURLOPT_POSTFIELDS: c_int = 10015;
pub const CURLOPT_POSTFIELDSIZE: c_int = 60;
pub const CURLOPT_SSL_VERIFYPEER: c_int = 64;
pub const CURLOPT_MAXFILESIZE: c_int = 114;
pub const CURLOPT_TIMEOUT_MS: c_int = 155;
// Patience guard (curl.h: CONNECTTIMEOUT=LONG+78) + the timeout result
// code (curlcode.h). Silence AFTER connect is policed by the app-level
// stall watchdog in rt/ai.rs, not by curl options.
pub const CURLOPT_CONNECTTIMEOUT: c_int = 78;
pub const CURLE_OPERATION_TIMEDOUT: c_int = 28;

pub const CURLINFO_EFFECTIVE_URL: c_int = 0x100001; /* STRING + 1 */
pub const CURLINFO_RESPONSE_CODE: c_int = 0x200002; /* LONG + 2 */
pub const CURLINFO_PRIVATE: c_int = 0x100015;       /* STRING + 21 (verified curl.h) */

pub type Curl = c_void;
pub type CurlM = c_void;

/// `struct curl_slist` — opaque linked list of "K: V" header strings.
#[repr(C)]
pub struct CurlSlist {
    _opaque: [u8; 0],
}

/// `CURLMsg` — { CURLMSG msg; CURL *easy_handle; union { CURLcode result;
/// void *ptr; } data; } (16 bytes; only the fields we read are mirrored).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CurlMsg {
    pub msg: c_int,
    _pad: [u8; 4],
    pub easy_handle: *mut Curl,
    pub result: c_int,
}

/// CURLOPT_WRITEFUNCTION / CURLOPT_HEADERFUNCTION signature.
pub type CurlWriteCallback = unsafe extern "C" fn(*mut c_void, usize, usize, *mut c_void) -> usize;

/// CURLMOPT_SOCKETFUNCTION signature:
/// int (*)(CURL *easy, curl_socket_t s, int action, void *userp, void *socketp)
pub type CurlSocketCallback = unsafe extern "C" fn(
    *mut Curl,
    c_int,
    c_int,
    *mut c_void,
    *mut c_void,
) -> c_int;

/// CURLMOPT_TIMERFUNCTION signature: int (*)(CURLM*, long timeout_ms, void*)
pub type CurlTimerCallback = unsafe extern "C" fn(*mut CurlM, c_long, *mut c_void) -> c_int;

extern "C" {
    pub fn curl_global_init(flags: c_long) -> c_int;
    pub fn curl_easy_init() -> *mut Curl;
    pub fn curl_easy_cleanup(curl: *mut Curl);
    pub fn curl_easy_strerror(code: c_int) -> *const c_char;
    pub fn curl_slist_append(list: *mut CurlSlist, s: *const c_char) -> *mut CurlSlist;
    pub fn curl_slist_free_all(list: *mut CurlSlist);
    pub fn curl_multi_init() -> *mut CurlM;
    pub fn curl_multi_add_handle(multi: *mut CurlM, easy: *mut Curl) -> c_int;
    pub fn curl_multi_remove_handle(multi: *mut CurlM, easy: *mut Curl) -> c_int;
    pub fn curl_multi_info_read(multi: *mut CurlM, msgs_in_queue: *mut c_int) -> *mut CurlMsg;
    pub fn curl_multi_socket_action(
        multi: *mut CurlM,
        s: c_int,
        ev_bitmask: c_int,
        running_handles: *mut c_int,
    ) -> c_int;
    pub fn curl_multi_assign(multi: *mut CurlM, sockfd: c_int, sockp: *mut c_void) -> c_int;
    pub fn curl_multi_strerror(code: c_int) -> *const c_char;
    // variadic option setters — call with the option's exact C vararg type.
    pub fn curl_easy_setopt(curl: *mut Curl, option: c_int, ...) -> c_int;
    pub fn curl_easy_getinfo(curl: *mut Curl, info: c_int, ...) -> c_int;
    pub fn curl_easy_perform(curl: *mut Curl) -> c_int;
    pub fn curl_multi_setopt(multi: *mut CurlM, option: c_int, ...) -> c_int;
}
