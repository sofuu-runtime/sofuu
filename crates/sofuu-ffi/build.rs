// build.rs — compiles the Sofuu C core (QuickJS + src/*.c + libuv + curl)
// into static libraries that the Rust binary links against.
//
// This is Phase 0: we compile the *same* C sources the Makefile uses, so the
// Rust entrypoint is a drop-in for the C main(). The C sources stay the
// source of truth until each subsystem is ported to Rust (see ROADMAP Track D).

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = root.parent().unwrap().parent().unwrap(); // crates/sofuu-ffi -> repo root
    let deps = root.join("deps");
    let src = root.join("src");

    // ── QTSQ codec (proprietary, latest checkout) ────────────────
    // Session data persists as .qtsq files, so the default build links the
    // QTSQ static lib. Point at a different checkout with SOFUU_QTSQ_DIR.
    let qtsq_dir = PathBuf::from(
        env::var("SOFUU_QTSQ_DIR")
            .unwrap_or_else(|_| "/Users/priyanshuboruah/projects/black-hole-disk".to_string()),
    );
    let qtsq_lib = qtsq_dir.join("libqtsq.a");
    let has_qtsq = qtsq_lib.exists();

    // ── Compiler flags ──────────────────────────────────────────
    // M10: the only C left in this build is the engine layer itself —
    // QuickJS, the SIMD kernels, and the vendored http-parser. Every
    // src/*.c glue file is retired (see the source list below). The
    // pre-M10 defines (SOFUU_MEMORY / SOFUU_RUST_CORE / SOFUU_QTSQ_PRESENT)
    // existed for engine.c / ffi_shim.c and are gone with them; the Rust
    // side carries the same facts via cargo cfgs (has_qtsq).
    let mut cc = cc::Build::new();
    cc.opt_level(2)
        .warnings(false)
        .include(&deps.join("quickjs"))
        .include(&deps.join("libuv").join("include"))
        .include(&src)
        .include(&src.join("simd"))
        .include(&deps.join("http-parser"))
        .define("_GNU_SOURCE", None)
        .define("CONFIG_VERSION", "\"2024-01-13\"")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-cast-function-type");

    // SIMD: pick the right kernel for the host arch.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "aarch64" {
        cc.flag_if_supported("-march=armv8-a");
    } else if target_arch == "x86_64" {
        cc.flag_if_supported("-mavx2");
        cc.flag_if_supported("-mfma");
    }

    // ── QuickJS sources ─────────────────────────────────────────
    cc.file(deps.join("quickjs/quickjs.c"));
    cc.file(deps.join("quickjs/quickjs-libc.c"));
    cc.file(deps.join("quickjs/libregexp.c"));
    cc.file(deps.join("quickjs/libunicode.c"));
    cc.file(deps.join("quickjs/cutils.c"));
    cc.file(deps.join("quickjs/libbf.c"));

    // ── Runtime C sources ───────────────────────────────────────
    // M10: only the SIMD kernels + the vendored http-parser remain. The
    // whole glue layer is retired — sofuu.c/engine.c/repl.c/ffi_shim.c
    // (M10) → crates/sofuu-core/src/rt/engine.rs, io/tui.c (M10) →
    // rt/tui.rs, all io/modules/http/mcp/npm/memory shells in M1–M9 →
    // their rt/*.rs twins. main.c is NOT (and never was) compiled here —
    // the Rust binary provides main().
    let sofuu_sources = [
        "simd/neon.c", // both compiled; each self-guards with #ifdef
        "simd/avx.c",
        "../deps/http-parser/http_parser.c",
    ];
    for s in &sofuu_sources {
        cc.file(src.join(s));
    }

    // ── Build static libsofuu_core.a ────────────────────────────
    cc.compile("sofuu_core");
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // ── Link the C runtime's dependencies ───────────────────────
    // libuv: prebuilt at deps/libuv/build/libuv.a (host), or a custom
    // directory via SOFUU_UV_DIR (scripts/cross builds into
    // deps/libuv/build-linux-<arch>).
    let uv_lib = match env::var("SOFUU_UV_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => root.join("deps/libuv/build"),
    };
    if uv_lib.join("libuv.a").exists() {
        println!("cargo:rustc-link-search=native={}", uv_lib.display());
        println!("cargo:rustc-link-lib=static=uv");
    } else {
        // Not prebuilt — tell the user to build it.
        panic!(
            "libuv not built: {}. Build it first \
             (deps/libuv: cmake -S . -B build && cmake --build build).",
            uv_lib.display()
        );
    }

    // curl: system lib on macOS (dynamic); the cross scripts build a
    // static, TLS-free libcurl at dist/libcurl-<target> and point here
    // with SOFUU_CURL_STATIC_DIR.
    match env::var("SOFUU_CURL_STATIC_DIR") {
        Ok(dir) => {
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:rustc-link-lib=static=curl");
        }
        Err(_) => println!("cargo:rustc-link-lib=curl"),
    }
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");

    // ── QTSQ static lib (session store + memory integration) ─────
    // Proprietary / local checkout. CI and third-party builds don't have it:
    // degrade gracefully (no session store / memory persistence) instead of
    // failing the build. Set SOFUU_QTSQ_DIR to a checkout to enable it.
    if has_qtsq {
        println!("cargo:rustc-link-search=native={}", qtsq_dir.display());
        println!("cargo:rustc-link-lib=static=qtsq");
        // The checkout's libqtsq.a references the qtc compression core
        // (qtc_decompress_*) that lives in its own archive — link it
        // AFTER qtsq so the static resolver sees the definitions.
        let qtc_lib = qtsq_dir.join("compressor/libqtc.a");
        if qtc_lib.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                qtsq_dir.join("compressor").display()
            );
            println!("cargo:rustc-link-lib=static=qtc");
            println!("cargo:rerun-if-changed={}", qtc_lib.display());
        }
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-check-cfg=cfg(has_qtsq)");
        println!("cargo:rustc-cfg=has_qtsq");
        // Rebuild when the checkout's artifacts change — without these a
        // refreshed libqtsq.a silently keeps the stale probe cfg and (until
        // some downstream crate edits) the stale link inputs.
        println!("cargo:rerun-if-changed={}", qtsq_lib.display());
        println!("cargo:rerun-if-changed={}", qtsq_dir.join("include/qtsq_format.h").display());
        println!("cargo:rerun-if-changed={}", qtsq_dir.join("include/qtsq.h").display());
    } else {
        println!(
            "cargo:warning=libqtsq.a not found at {} — building WITHOUT QTSQ \
             (session store + brain persistence disabled). Set SOFUU_QTSQ_DIR \
             to a QTSQ checkout to enable them.",
            qtsq_dir.display()
        );
    }

    // macOS: curl may need these extra frameworks. The QTSQ static lib also
    // embeds ObjC++/Metal/CoreBluetooth surfaces, so we add those too.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreBluetooth");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=objc");
    }
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_DIR");

    // ── libopus (optional, dev/test links) ─────────────────────────
    // The QTSQ archive's audio objects reference libopus. Release links
    // (LTO) prune those objects, but debug/test links pull them → link
    // fails without it. Add it when pkg-config finds it — only in QTSQ
    // builds (a host pkg-config result would poison a cross build).
    if has_qtsq {
        if let Ok(out) = std::process::Command::new("pkg-config")
            .args(["--libs", "opus"])
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                for tok in s.split_whitespace() {
                    if let Some(lib) = tok.strip_prefix("-l") {
                        println!("cargo:rustc-link-lib={}", lib);
                    } else if let Some(dir) = tok.strip_prefix("-L") {
                        println!("cargo:rustc-link-search=native={}", dir);
                    }
                }
            }
        }
    }

    // ── QTSQ layout guard (regression) ────────────────────────────
    // Compiles a C file of _Static_asserts against the checkout's
    // qtsq_format.h — syntax-only, nothing is executed (arg-list Command,
    // no shell), so cross builds are covered too. A checkout that changes
    // sizeof/offsets fails the BUILD here instead of silently corrupting
    // brain files at runtime. Bump these expected values together with
    // the Rust mirrors in crates/sofuu-ffi/src/qtsq.rs.
    if has_qtsq {
        let guard_src = out_dir.join("qtsq_layout_guard.c");
        std::fs::write(
            &guard_src,
            r#"#include <stddef.h>
#include "qtsq_format.h"
_Static_assert(sizeof(qtsq_context_t) == 5024, "QTSQ_CONTEXT_SIZE drifted");
_Static_assert(offsetof(qtsq_context_t, header.data_type) == 10, "OFF_HEADER_DATA_TYPE drifted");
_Static_assert(offsetof(qtsq_context_t, schema) == 112, "OFF_SCHEMA drifted");
_Static_assert(offsetof(qtsq_schema_t, dimensions) == 68, "OFF_SCHEMA_DIMENSIONS drifted");
_Static_assert(offsetof(qtsq_schema_t, num_dims) == 100, "OFF_SCHEMA_NUM_DIMS drifted");
_Static_assert(offsetof(qtsq_context_t, is_encrypted) == 4908, "OFF_IS_ENCRYPTED drifted");
"#,
        )
        .expect("write qtsq layout guard");
        let compiled = std::process::Command::new("cc")
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(qtsq_dir.join("include"))
            .arg(&guard_src)
            .output();
        let ok = compiled.as_ref().map(|o| o.status.success()).unwrap_or(false);
        if !ok {
            let stderr = compiled
                .map(|o| String::from_utf8_lossy(&o.stderr).into_owned())
                .unwrap_or_else(|e| e.to_string());
            panic!(
                "QTSQ layout guard failed: the checkout's qtsq_context_t no \
                 longer matches the Rust mirrors in qtsq.rs (sizeof=5024, \
                 offsets 10/112/68/100/4908). Update the mirrors and this \
                 guard together.\n{stderr}"
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", src.display());
}

fn _unused(_p: &Path) {}
