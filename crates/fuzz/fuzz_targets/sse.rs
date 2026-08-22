//! cargo-fuzz target for the Rust SSE parser (sofuu_core::http::sse).
//!
//! Invocation (from the repo root):
//!   cargo install cargo-fuzz
//!   cargo +nightly fuzz run sse -- -max_len=4096 -timeout=5
//! The Cargo.toml at crates/fuzz is a standalone workspace — run from there:
//!   cd crates/fuzz && cargo fuzz run sse -- -max_len=4096

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The parser is lossy-decoding by design — any bytes are safe input.
    let text = String::from_utf8_lossy(data);
    let mut parser = sofuu_core::http::sse::SseParser::new();
    let events = parser.feed(&text);
    // Events must never panic; exercise their fields.
    for e in &events {
        let _ = e.event.len() + e.data.len();
    }
    // A second feed with the remainder must be safe too (state machine reuse).
    let _ = parser.feed("data: tail\n\n");
});
