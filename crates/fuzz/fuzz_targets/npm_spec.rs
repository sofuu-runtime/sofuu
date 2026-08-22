//! cargo-fuzz target for the npm spec validator (sofuu_core::npm::spec_is_safe).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    // spec_is_safe must never panic on arbitrary input.
    let _ = sofuu_core::npm::spec_is_safe(&text);
    // split_spec + clean_version are also pure string logic.
    let (name, ver) = sofuu_core::npm::split_spec(&text);
    let _ = sofuu_core::npm::clean_version(&ver);
    let _ = name.len();
});
