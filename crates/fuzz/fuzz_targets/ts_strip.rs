//! cargo-fuzz target for the Rust TypeScript stripper (sofuu_core::ts::strip).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    // strip() must never panic on arbitrary input; the output must be
    // valid UTF-8 (it's a String).
    let out = sofuu_core::ts::strip(&text);
    debug_assert!(out.is_char_boundary(0));
    let _ = out.len();
});
