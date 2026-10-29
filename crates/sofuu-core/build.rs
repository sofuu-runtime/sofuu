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
    // The whitelisted experiment widths are emitted as cfgs below; declare
    // them so setting SOFUU_EMB_H doesn't trip the unexpected_cfgs lint.
    println!("cargo:rustc-check-cfg=cfg(emb_h_32,emb_h_36,emb_h_48,emb_h_64,emb_h_96,emb_h_128)");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";
    // R-22 (AUDIT-2026-10-29): the probe used to end in a hardcoded absolute
    // developer checkout path, which made the source non-portable and put an
    // individual engineer's home directory in the repository. Resolution is
    // now, in order:
    //   1. SOFUU_QTSQ_DIR (the documented, CI-safe mechanism);
    //   2. $HOME/projects/black-hole-disk — the same checkout expressed
    //      relatively, so a developer box keeps working with no setup while
    //      no absolute username path is compiled into the tree;
    //   3. a vendored deps/qtsq inside this repository, if one is added later.
    // A missing library compiles the memory module out exactly as before.
    let qtsq_dir = env::var("SOFUU_QTSQ_DIR")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.join("libqtsq.a").exists())
        .or_else(|| {
            let home = env::var("HOME").ok()?;
            let candidate = PathBuf::from(home).join("projects").join("black-hole-disk");
            if candidate.join("libqtsq.a").exists() {
                Some(candidate)
            } else {
                None
            }
        })
        .or_else(|| {
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
            let candidate = manifest.join("../../deps/qtsq");
            if candidate.join("libqtsq.a").exists() {
                Some(candidate)
            } else {
                None
            }
        });
    // P3 (AUDIT-2026-09-07): the probe used to look for libqtsq.a on every
    // host while sofuu-ffi/build.rs links qtsq.lib (or SOFUU_QTSQ_LIB) on
    // Windows — the two crates' has_qtsq cfgs could disagree on a Windows
    // MSVC layout, compiling the memory module out while the FFI crate
    // still links QTSQ. Mirror the FFI resolution exactly.
    let qtsq_lib = match &qtsq_dir {
        Some(dir) if is_windows => env::var("SOFUU_QTSQ_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dir.join("windows/build-win64/Release/qtsq.lib")),
        Some(dir) => dir.join("libqtsq.a"),
        None => PathBuf::new(),
    };
    let has_qtsq = qtsq_dir.is_some() && qtsq_lib.exists();
    // B7 (ROADMAP-BRAIN-CMA): watch the probed paths on EVERY outcome —
    // emitting them only on success let a failed probe cache forever (the
    // codec owner rebuilds libqtsq.a, cargo never re-probes, and the build
    // stays silently memory-less until a manual `touch build.rs`).
    if let Some(dir) = &qtsq_dir {
        println!("cargo:rerun-if-changed={}", qtsq_lib.display());
        println!("cargo:rerun-if-changed={}", dir.join("include/qtsq_format.h").display());
        println!("cargo:rerun-if-changed={}", dir.join("include/qtsq.h").display());
    }
    if has_qtsq {
        println!("cargo:rustc-cfg=has_qtsq");
    } else if let Some(dir) = &qtsq_dir {
        panic!(
            "{} not found at {} — QTSQ is a mandatory core component of Sofuu. \
             Set SOFUU_QTSQ_DIR to a built QTSQ checkout (SOFUU_QTSQ_LIB overrides the archive path on Windows).",
            qtsq_lib.file_name().and_then(|n| n.to_str()).unwrap_or("QTSQ archive"),
            dir.display()
        );
    } else {
        panic!(
            "SOFUU_QTSQ_DIR not set — QTSQ is a mandatory core component of Sofuu. \
             Set it to a built QTSQ checkout to build Sofuu."
        );
    }
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_DIR");
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_LIB");
    // hidden-width experiments: whitelist + emit emb_h_* cfgs
    match std::env::var("SOFUU_EMB_H").as_deref() {
        Ok("32") => println!("cargo:rustc-cfg=emb_h_32"),
        Ok("36") => println!("cargo:rustc-cfg=emb_h_36"),
        Ok("48") => println!("cargo:rustc-cfg=emb_h_48"),
        Ok("64") => println!("cargo:rustc-cfg=emb_h_64"),
        Ok("96") => println!("cargo:rustc-cfg=emb_h_96"),
        Ok("128") => println!("cargo:rustc-cfg=emb_h_128"),
        // P3 (AUDIT-2026-09-07): anything else used to be swallowed — a typo
        // like SOFUU_EMB_H=16 silently built the default width and the
        // experiment never happened. Make the rejection visible.
        Ok(other) => println!(
            "cargo:warning=SOFUU_EMB_H={other} is not a whitelisted width \
             (32/36/48/64/96/128) — building with the default H. \
             See the ml embedding model for the supported set."
        ),
        Err(_) => {}
    }
    println!("cargo:rerun-if-env-changed=SOFUU_EMB_H");
    // Prefill Phase 3: AMX via Accelerate's cblas_sgemm (macOS only; the NEON
    // tile path stays as fallback and Windows/Linux never link the framework).
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rerun-if-changed=build.rs");
    // M1: stage the trained image-projector artifact (weights_img.sem) into
    // OUT_DIR for include_bytes!. Absent until first training → stage an
    // empty blob; the runtime treats it as model-unavailable (tested), never
    // as silent garbage. Rebuilds when the trainer refreshes the artifact.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let staged = out_dir.join("weights_img.sem");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let baked = manifest.join("src/embedding/weights_img.sem");
    println!("cargo:rerun-if-changed={}", baked.display());
    let bytes = std::fs::read(&baked).unwrap_or_default();
    if bytes.is_empty() && baked.exists() {
        println!("cargo:warning=weights_img.sem is empty — image embeddings report model-unavailable until trained");
    }
    let _ = std::fs::write(&staged, bytes);
}
