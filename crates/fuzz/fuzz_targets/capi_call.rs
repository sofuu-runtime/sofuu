//! cargo-fuzz target for the sofuu_rt_call funnel (PLAN-HEADLESS H6).
//!
//! `sofuu_rt_call` with arbitrary method/JSON must never crash — a crash or
//! exit is a failed fuzz run. This exercises the full path: runtime init,
//! funnel dispatch, JSON parse, and error envelope generation.
//!
//! Invocation (from crates/fuzz):
//!   cargo fuzz run capi_call -- -max_len=256 -timeout=5
//!
//! NOTE: this target links sofuu-capi, which links the full engine. It's
//! heavier than the pure-parser targets — expect slower throughput.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::{CStr, CString};

// We link against the static capi library. The symbols are declared in
// include/sofuu_embed.h.
extern "C" {
    fn sofuu_rt_new(config_json: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
    fn sofuu_rt_free(rt: *mut std::ffi::c_void);
    fn sofuu_rt_call(
        rt: *mut std::ffi::c_void,
        method: *const std::os::raw::c_char,
        args_json: *const std::os::raw::c_char,
        out_json: *mut *mut std::os::raw::c_char,
    ) -> std::os::raw::c_int;
    fn sofuu_free(ptr: *mut std::ffi::c_void);
}

/// Split fuzzer input into a method string and an args JSON string.
/// We use a simple split: the first NUL byte (or a fixed-length prefix)
/// separates method from args. If no separator, treat the whole input as method.
fn split_input(data: &[u8]) -> (String, String) {
    // Use the byte 0xFF as separator (it won't appear in valid UTF-8 method names).
    if let Some(pos) = data.iter().position(|&b| b == 0xFF) {
        let method = String::from_utf8_lossy(&data[..pos]).to_string();
        let args = String::from_utf8_lossy(&data[pos + 1..]).to_string();
        (method, args)
    } else {
        (String::from_utf8_lossy(data).to_string(), String::new())
    }
}

fuzz_target!(|data: &[u8]| {
    // Fuzz input is arbitrary bytes — split into method + args.
    let (method, args) = split_input(data);

    // Skip empty methods (trivially handled).
    if method.is_empty() {
        return;
    }

    // Create a runtime (QTSQ-free, no config).
    let rt = unsafe { sofuu_rt_new(std::ptr::null()) };
    if rt.is_null() {
        return; // init failed — acceptable, not a crash
    }

    // Call the funnel with arbitrary method + args.
    let method_c = match CString::new(method.as_str()) {
        Ok(c) => c,
        Err(_) => {
            unsafe { sofuu_rt_free(rt) };
            return;
        }
    };
    let args_c = match CString::new(args.as_str()) {
        Ok(c) => c,
        Err(_) => {
            unsafe { sofuu_rt_free(rt) };
            return;
        }
    };

    let mut out_json: *mut std::os::raw::c_char = std::ptr::null_mut();
    let _rc = unsafe {
        sofuu_rt_call(
            rt,
            method_c.as_ptr(),
            args_c.as_ptr(),
            &mut out_json,
        )
    };

    // Free the output if allocated.
    if !out_json.is_null() {
        unsafe { sofuu_free(out_json as *mut _) };
    }

    // Clean up — must not crash or hang.
    unsafe { sofuu_rt_free(rt) };
});
