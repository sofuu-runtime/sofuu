// build.rs — mirrors crates/sofuu-ffi/build.rs's QTSQ probe for sofuu-core.
//
// `cargo:rustc-cfg` emitted by a build script applies only to that script's
// own crate, so the two crates probe independently. rt/memory.rs uses
// `#[cfg(has_qtsq)]` to keep the QTSQ-dependent memory shells (brain-file +
// KV page persistence) compiled only when the vendored libqtsq.a checkout is
// available — the same condition the C build uses for SOFUU_QTSQ_PRESENT.
// Without it the `mod_*_register` exports exist as no-ops and engine.c (which
// only calls them under its own QTSQ guard) sees the same JS-visible absence.

use std::env;
use std::path::PathBuf;

fn main() {
    // Declare the cfg so the `unexpected_cfgs` lint stays quiet in both
    // configurations (see Cargo's check-cfg docs).
    println!("cargo:rustc-check-cfg=cfg(has_qtsq)");
    let qtsq_dir = PathBuf::from(
        env::var("SOFUU_QTSQ_DIR")
            .unwrap_or_else(|_| "/Users/priyanshuboruah/projects/black-hole-disk".to_string()),
    );
    if qtsq_dir.join("libqtsq.a").exists() {
        println!("cargo:rustc-cfg=has_qtsq");
    } else {
        println!(
            "cargo:warning=libqtsq.a not found at {} — building WITHOUT QTSQ \
             (memory module registers as no-op). Set SOFUU_QTSQ_DIR to a QTSQ \
             checkout to enable brain persistence.",
            qtsq_dir.display()
        );
    }
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}
