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
// P3 (AUDIT-2026-09-07): fetch stall detection + thread-safe DNS — values
// verified against the macOS SDK curl.h (LOW_SPEED_LIMIT=LONG+19,
// LOW_SPEED_TIME=LONG+20, NOSIGNAL=LONG+99, CONNECTTIMEOUT_MS=LONG+156).
pub const CURLOPT_CONNECTTIMEOUT_MS: c_int = 156;
pub const CURLOPT_LOW_SPEED_LIMIT: c_int = 19;
pub const CURLOPT_LOW_SPEED_TIME: c_int = 20;
pub const CURLOPT_NOSIGNAL: c_int = 99;
pub const CURLE_OPERATION_TIMEDOUT: c_int = 28;
// Redirect hardening (curl.h): only http/https on redirect, cap hops,
// never forward Authorization cross-host.
pub const CURLOPT_REDIR_PROTOCOLS: c_int = 182;
/* proc-7 (AUDIT-2026-09-07): the same lock for the FIRST request — without
 * it a non-https URL accepted before any redirect could serve attacker
 * bytes (the redirect lock never gets a chance to apply). */
pub const CURLOPT_PROTOCOLS: c_int = 181;
pub const CURLOPT_MAXREDIRS: c_int = 68;
pub const CURLOPT_UNRESTRICTED_AUTH: c_int = 105;
pub const CURLPROTO_HTTP: c_long = 1;
pub const CURLPROTO_HTTPS: c_long = 2;
// net-4 (AUDIT-2026-09-07) wire-level address guard — verified with a
// compiled probe against this SDK (curl.h): OPENSOCKETFUNCTION is
// CURLOPTTYPE_FUNCTIONPOINT(20000)+163 = 20163 (NOT 20140); OPENSOCKETDATA
// is OBJECTPOINT(10000)+164. Returning CURL_SOCKET_BAD aborts the
// connection with CURLE_COULDNT_CONNECT — libcurl's documented mechanism
// for IP address block-listing.
pub const CURLOPT_OPENSOCKETFUNCTION: c_int = 20163;
pub const CURLOPT_OPENSOCKETDATA: c_int = 10164;
pub const CURL_SOCKET_BAD: c_int = -1;
pub const CURLSOCKTYPE_IPCXN: c_int = 0;
pub const CURLE_COULDNT_CONNECT: c_int = 7;

pub const CURLINFO_EFFECTIVE_URL: c_int = 0x100001; /* STRING + 1 */
pub const CURLINFO_RESPONSE_CODE: c_int = 0x200002; /* LONG + 2 */
pub const CURLINFO_PRIVATE: c_int = 0x100015;       /* STRING + 21 (verified curl.h) */

pub type Curl = c_void;
pub type CurlM = c_void;
// M2: opaque libcurl mime types (multipart bodies for audio upload).
pub type CurlMime = c_void;
pub type CurlMimePart = c_void;
// CURLOPT_MIMEPOST = OBJECTPOINT(10000) + 269 (verified curl.h).
pub const CURLOPT_MIMEPOST: c_int = 10269;

/// `struct curl_slist` — opaque linked list of "K: V" header strings.
#[repr(C)]
pub struct CurlSlist {
    _opaque: [u8; 0],
}

/// `CURLMsg` — { CURLMSG msg; CURL *easy_handle; union { CURLcode result;
/// void *ptr; } data; } (24 bytes on 64-bit — msg + 4 pad + handle +
/// 8-byte union; the mirror reads `result` in place of the union and keeps
/// the C ABI layout via repr(C) padding).
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

/// `struct curl_sockaddr` (curl.h:416) — the address libcurl proposes to
/// connect to, handed to CURLOPT_OPENSOCKETFUNCTION. The trailing `addr`
/// member is `struct sockaddr` (16 bytes) in the public struct, but libcurl
/// passes a larger internal `Curl_sockaddr_ex` and the man contract lets the
/// callback read `addrlen` bytes at `addr` — sockaddr_in6 needs 28. The
/// mirror therefore over-allocates the tail and all reads go through
/// `addrlen`-checked slices, never a deref of a C `sockaddr_in*` type.
#[repr(C)]
pub struct CurlSockaddr {
    pub family: c_int,
    pub socktype: c_int,
    pub protocol: c_int,
    pub addrlen: c_int, /* unsigned int in C — read as i32, values are tiny */
    pub addr: [u8; 128], /* over-sized tail: matches Curl_sockaddr_ex usage */
}

/// CURLOPT_OPENSOCKETFUNCTION signature:
/// curl_socket_t (*)(void *clientp, curlsocktype purpose, struct curl_sockaddr *address)
pub type CurlOpenSocketCallback =
    unsafe extern "C" fn(*mut c_void, c_int, *mut CurlSockaddr) -> c_int;

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
    // M2: mime API for multipart/form-data (audio transcription upload).
    // curl_mime_data copies the payload; the req Box still owns the bytes.
    pub fn curl_mime_init(easy: *mut Curl) -> *mut CurlMime;
    pub fn curl_mime_free(mime: *mut CurlMime);
    pub fn curl_mime_addpart(mime: *mut CurlMime) -> *mut CurlMimePart;
    pub fn curl_mime_name(part: *mut CurlMimePart, name: *const c_char) -> c_int;
    pub fn curl_mime_filename(part: *mut CurlMimePart, filename: *const c_char) -> c_int;
    pub fn curl_mime_type(part: *mut CurlMimePart, mimetype: *const c_char) -> c_int;
    pub fn curl_mime_data(part: *mut CurlMimePart, data: *const c_char, datasize: usize) -> c_int;
}
