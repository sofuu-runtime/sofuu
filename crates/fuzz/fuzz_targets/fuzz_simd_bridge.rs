//! cargo-fuzz target for the validated JS-view window (rt/view_guard.rs).
//!
//! The typed-array bridge hands native kernels (base, byte_offset, byte_length)
//! taken from a JavaScript `Float32Array`. `check_f32_view` is the function
//! that re-derives the invariant — aligned, in bounds, optional exact
//! dimension — before any pointer is used. This target feeds it arbitrary
//! offsets/lengths/buffer sizes, including the wrap-around cases a malicious
//! or buggy caller would produce, and asserts two things on every input:
//!
//!   1. no panic, no overflow, no allocation surprise (the function is pure
//!      arithmetic), and
//!   2. an ACCEPTED window really is a subset of the buffer, so the invariant
//!      the kernels rely on cannot be violated by construction.
//!
//! Invocation (from crates/fuzz):
//!   cargo +nightly fuzz run fuzz_simd_bridge -- -max_len=4096 -timeout=5
//!
//! No QuickJS context is needed per input: the arithmetic is the boundary.
//! The JS-facing behaviour (detached buffers, wrong TypedArray class) is
//! covered by the JS suite and the Rust unit tests.

#![no_main]

use libfuzzer_sys::fuzz_target;

/// Decode `data` as three little-endian u64s, shrinking to platform usize
/// with the top bits dropped (the fuzzer only needs coverage of the space).
fn take(data: &[u8], at: usize) -> usize {
    let mut b = [0u8; 8];
    if let Some(s) = data.get(at..at + 8) {
        b.copy_from_slice(s);
    }
    u64::from_le_bytes(b) as usize
}

fuzz_target!(|data: &[u8]| {
    let buf_size = take(data, 0);
    let byte_offset = take(data, 8);
    let byte_length = take(data, 16);
    let selector = data.get(24).copied().unwrap_or(0);
    let expected = if selector & 1 == 1 {
        None
    } else {
        // Half the runs pin a dimension, including 0 and huge values.
        Some((selector >> 1) as usize)
    };

    match sofuu_core::rt::view_guard::check_f32_view(
        buf_size,
        byte_offset,
        byte_length,
        expected,
    ) {
        Ok(w) => {
            // (2) The accepted window is a whole, in-bounds f32 range.
            assert_eq!(w.offset % 4, 0, "accepted an unaligned offset");
            assert_eq!(w.count, byte_length / 4, "count must match the length");
            assert_eq!(w.offset, byte_offset, "offset must be preserved verbatim");
            let end = w
                .offset
                .checked_add(w.count * 4)
                .expect("accepted window must not overflow");
            assert!(end <= buf_size, "accepted a window past the buffer end");
            if let Some(want) = expected {
                assert_eq!(w.count, want, "accepted a dimension mismatch");
            }
        }
        Err(_) => {
            // Refusals are total: either the window was unaligned, out of
            // bounds, or off-dimension. Nothing else may escape.
        }
    }

    // The byte form must agree with the lane form for 4-byte units.
    let _ = sofuu_core::rt::view_guard::check_view(1, buf_size, byte_offset, byte_length, None);
    let _ = sofuu_core::rt::view_guard::check_view(4, buf_size, byte_offset, byte_length, expected);
});
