// sofuu-ffi — vendored http-parser bindings (PLAN-RUST-MIGRATION M5).
//
// The HTTP server port drives node's legacy http_parser (deps/http-parser)
// the same way the retired src/http/server.c did. Only the req-side surface
// is used: init/execute/method_str + the on_url/on_body/on_message_complete
// callbacks. The parser struct is mirrored from http_parser.h:297-330 —
// NATURAL alignment (no packed attribute): 32 bytes on 64-bit.
//
//   bits0:  type:2 | flags:8 | state:7 | header_state:7 | index:5 |
//           uses_transfer_encoding:1 | allow_chunked_length:1 |
//           lenient_http_headers:1            (32 bits)
//   nread: u32                       offset 4
//   content_length: u64              offset 8
//   http_major/http_minor: u16,u16   offset 16
//   bits1:  status_code:16 | method:8 | http_errno:7 | upgrade:1
//                                     offset 20
//   data: *mut c_void                 offset 24

use std::ffi::c_char;
use std::os::raw::c_int;
use std::os::raw::c_void;

pub const HTTP_REQUEST: c_int = 0;
pub const HPE_OK: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HttpParser {
    pub bits0: u32,
    pub nread: u32,
    pub content_length: u64,
    pub http_major: u16,
    pub http_minor: u16,
    pub bits1: u32,
    pub data: *mut c_void,
}

impl HttpParser {
    /// `(p)->method` — requests only (bits1 >> 16) & 0xFF.
    #[inline]
    pub fn method(&self) -> c_int {
        ((self.bits1 >> 16) & 0xFF) as c_int
    }

    /// `(p)->http_errno` — (bits1 >> 24) & 0x7F.
    #[inline]
    pub fn http_errno(&self) -> c_int {
        ((self.bits1 >> 24) & 0x7F) as c_int
    }
}

/// `http_cb` — int (*)(http_parser*).
pub type HttpCb = unsafe extern "C" fn(*mut HttpParser) -> c_int;
/// `http_data_cb` — int (*)(http_parser*, const char*, size_t).
pub type HttpDataCb = unsafe extern "C" fn(*mut HttpParser, *const c_char, usize) -> c_int;

/// `struct http_parser_settings` — 10 callbacks (80 bytes on 64-bit).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HttpParserSettings {
    pub on_message_begin: Option<HttpCb>,
    pub on_url: Option<HttpDataCb>,
    pub on_status: Option<HttpDataCb>,
    pub on_header_field: Option<HttpDataCb>,
    pub on_header_value: Option<HttpDataCb>,
    pub on_headers_complete: Option<HttpCb>,
    pub on_body: Option<HttpDataCb>,
    pub on_message_complete: Option<HttpCb>,
    pub on_chunk_header: Option<HttpCb>,
    pub on_chunk_complete: Option<HttpCb>,
}

extern "C" {
    pub fn http_parser_init(parser: *mut HttpParser, type_: c_int);
    pub fn http_parser_execute(
        parser: *mut HttpParser,
        settings: *const HttpParserSettings,
        data: *const c_char,
        len: usize,
    ) -> usize;
    pub fn http_method_str(method: c_int) -> *const c_char;
}
