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

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_msvc = target_env == "msvc";
    let is_windows = target_os == "windows";

    // ── QTSQ codec (proprietary, latest checkout) ────────────────
    // Session data persists as .qtsq files, so the default build links the
    // QTSQ static lib. Point at a different checkout with SOFUU_QTSQ_DIR.
    // POSIX builds link the Makefile product libqtsq.a; Windows links the
    // MSVC port built by <checkout>/windows/CMakeLists.txt — a SINGLE
    // archive (qtsq.lib) that bakes in the qtc compressor, the crypto
    // suite, and the windows/shim POSIX-compat layer. SOFUU_QTSQ_LIB
    // overrides the archive path.
    // P2-21 (AUDIT-2026-09-01): no developer-machine absolute path as the
    // default (it leaked a private checkout name in a public repo and
    // broke every other machine anyway). SOFUU_QTSQ_DIR is REQUIRED when
    // the QTSQ feature is enabled; without the env var the build proceeds
    // as the no-QTSQ codec-less shape exactly like a missing checkout.
    let qtsq_dir = env::var("SOFUU_QTSQ_DIR").map(PathBuf::from).ok();
    let (qtsq_lib, qtsq_search_dir) = match &qtsq_dir {
        Some(dir) if is_windows => {
            let lib = env::var("SOFUU_QTSQ_LIB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.join("windows/build-win64/Release/qtsq.lib"));
            let search = lib.parent().map(Path::to_path_buf).unwrap_or_else(|| dir.clone());
            (lib, search)
        }
        Some(dir) => (dir.join("libqtsq.a"), dir.clone()),
        /* No SOFUU_QTSQ_DIR: the build proceeds as the documented
         * codec-less shape (has_qtsq = false). */
        None => (PathBuf::new(), PathBuf::new()),
    };
    // has_qtsq can only be true when qtsq_dir is Some — the gated blocks
    // below unwrap on exactly that invariant.
    let has_qtsq = qtsq_dir.is_some() && qtsq_lib.exists();

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

    // MSVC: the vendored C uses POSIX headers QuickJS's own _WIN32 branches
    // don't fully cover (unistd.h, sys/time.h, dirent.h). A compat include
    // dir provides them; only visible to cl.exe, gcc/clang never see it.
    if is_msvc {
        cc.include(&deps.join("quickjs/msvc-compat"));
    }

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
    // deps/libuv/build-linux-<arch>; Windows CI points it at the
    // cmake build dir containing uv_a.lib).
    let uv_lib = match env::var("SOFUU_UV_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => root.join("deps/libuv/build"),
    };
    if is_windows {
        // MSVC does no lib-prefixing and the CMake MSVC build produces
        // uv_a.lib (BUILD_SHARED_LIBS=OFF), not libuv.a. The zig/mingw
        // cross build (scripts/cross/build_windows_x86_64.sh) produces
        // the same `uv_a` target as a unix-style archive: libuv_a.a.
        let candidates: &[(&str, &str)] = if is_msvc {
            &[("uv_a.lib", "uv_a")]
        } else {
            &[
                ("libuv_a.a", "uv_a"),
                ("uv_a.a", "uv_a"),
                ("libuv.a", "uv"),
            ]
        };
        if let Some((_file, link)) = candidates
            .iter()
            .find(|(f, _)| uv_lib.join(f).exists())
        {
            println!("cargo:rustc-link-search=native={}", uv_lib.display());
            println!("cargo:rustc-link-lib=static={link}");
        } else {
            panic!(
                "libuv not built for the Windows target: {} (no {} found). \
                 Build it first (deps/libuv: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
                 -DBUILD_SHARED_LIBS=OFF && cmake --build build --config Release) \
                 or point SOFUU_UV_DIR at the directory holding it.",
                uv_lib.display(),
                if is_msvc { "uv_a.lib" } else { "libuv_a.a" }
            );
        }
    } else if uv_lib.join("libuv.a").exists() {
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
    // with SOFUU_CURL_STATIC_DIR. On Windows the vcpkg static-MD triplet
    // (curl[schannel]) produces libcurl.lib — CI exports
    // VCPKG_INSTALLATION_ROOT and we probe it automatically.
    if is_windows {
        let mut linked = false;
        let probe = |dir: PathBuf, label: &str| {
            for name in ["libcurl.lib", "curl.lib", "libcurl.a"] {
                if dir.join(name).exists() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                    if name == "libcurl.a" {
                        // zig/mingw cross build (scripts/cross/
                        // build_windows_x86_64.sh) produces a unix-style
                        // static archive — link it as -lstatic=curl.
                        println!("cargo:rustc-link-lib=static=curl");
                    } else {
                        // MSVC: pass the import/static lib name verbatim.
                        let stem = name.trim_end_matches(".lib");
                        println!("cargo:rustc-link-lib={}", stem);
                    }
                    return true;
                }
            }
            eprintln!("curl: no libcurl.lib/libcurl.a in {} ({})", dir.display(), label);
            false
        };
        if let Ok(dir) = env::var("SOFUU_CURL_STATIC_DIR") {
            linked = probe(PathBuf::from(dir), "SOFUU_CURL_STATIC_DIR");
        }
        if !linked {
            if let Ok(vcpkg) = env::var("VCPKG_INSTALLATION_ROOT") {
                linked = probe(
                    PathBuf::from(vcpkg)
                        .join("installed")
                        .join("x64-windows-static-md")
                        .join("lib"),
                    "vcpkg x64-windows-static-md",
                );
            }
        }
        if !linked {
            panic!(
                "curl not found for Windows: install vcpkg curl[schannel] \
                 (triplet x64-windows-static-md) for MSVC, or point \
                 SOFUU_CURL_STATIC_DIR at a directory containing libcurl.lib \
                 (MSVC) or libcurl.a (zig/mingw cross)."
            );
        }
    } else {
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
    }

    if is_windows {
        // Winsock + the Win32 APIs libuv/quickjs-libc/curl[schannel] pull in.
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=iphlpapi");
        println!("cargo:rustc-link-lib=psapi");
        println!("cargo:rustc-link-lib=userenv");
        println!("cargo:rustc-link-lib=user32");
        println!("cargo:rustc-link-lib=dbghelp");
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=oleaut32");
        println!("cargo:rustc-link-lib=uuid");
        println!("cargo:rustc-link-lib=crypt32");
        println!("cargo:rustc-link-lib=secur32");
        println!("cargo:rustc-link-lib=ncrypt");
        println!("cargo:rustc-link-lib=bcrypt");
        println!("cargo:rustc-link-lib=normaliz");
        println!("cargo:rustc-link-lib=dnsapi");
    }

    // ── QTSQ static lib (session store + memory integration) ─────
    // Proprietary / local checkout. CI and third-party builds don't have it:
    // degrade gracefully (no session store / memory persistence) instead of
    // failing the build. Set SOFUU_QTSQ_DIR to a checkout to enable it.
    // Declared unconditionally so the cfg name is always known to rustc —
    // otherwise no-QTSQ builds warn unexpected_cfgs on every #[cfg(has_qtsq)].
    println!("cargo:rustc-check-cfg=cfg(has_qtsq)");
    if has_qtsq {
        let qtsq_dir = qtsq_dir.as_deref().expect("has_qtsq implies SOFUU_QTSQ_DIR");
        println!("cargo:rustc-link-search=native={}", qtsq_search_dir.display());
        println!("cargo:rustc-link-lib=static=qtsq");
        // The POSIX Makefile product is TWO archives: libqtsq.a references
        // the qtc compression core (qtc_decompress_*) that lives in its own
        // archive — link it AFTER qtsq so the static resolver sees the
        // definitions. The Windows CMake build bakes qtc INTO the single
        // qtsq.lib (windows/CMakeLists.txt), so there's no second archive.
        if !is_windows {
            let qtc_lib = qtsq_dir.join("compressor/libqtc.a");
            if qtc_lib.exists() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    qtsq_dir.join("compressor").display()
                );
                println!("cargo:rustc-link-lib=static=qtc");
                println!("cargo:rerun-if-changed={}", qtc_lib.display());
            }
        }
        // zlib: POSIX resolves -lz from the system toolchain. Windows has no
        // system zlib — probe the same vcpkg triplet the curl block uses
        // (the CMake port's find_package(ZLIB) satisfies it the same way),
        // with SOFUU_ZLIB_DIR as the manual override. bcrypt/advapi32 (the
        // two qtsq crypto providers) are already linked by the Windows
        // system-lib block above.
        if is_windows {
            let mut linked = false;
            let probe = |dir: PathBuf, label: &str| {
                if dir.join("zlib.lib").exists() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                    println!("cargo:rustc-link-lib=zlib");
                    true
                } else {
                    eprintln!("qtsq: no zlib.lib in {} ({})", dir.display(), label);
                    false
                }
            };
            if let Ok(dir) = env::var("SOFUU_ZLIB_DIR") {
                linked = probe(PathBuf::from(dir), "SOFUU_ZLIB_DIR");
            }
            if !linked {
                if let Ok(vcpkg) = env::var("VCPKG_INSTALLATION_ROOT") {
                    linked = probe(
                        PathBuf::from(vcpkg)
                            .join("installed")
                            .join("x64-windows-static-md")
                            .join("lib"),
                        "vcpkg x64-windows-static-md",
                    );
                }
            }
            if !linked {
                panic!(
                    "zlib not found for the QTSQ MSVC port: install vcpkg zlib \
                     (triplet x64-windows-static-md) or point SOFUU_ZLIB_DIR at \
                     a directory containing zlib.lib."
                );
            }
        } else if let Ok(dir) = env::var("SOFUU_ZLIB_DIR") {
            // Cross builds (static musl, zig toolchain) have no system
            // zlib: scripts/cross/build_qtsq_cross.sh vendors libz.a into
            // the per-target dist dir and points here.
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:rustc-link-lib=static=z");
        } else {
            println!("cargo:rustc-link-lib=z");
        }
        // has_qtsq check-cfg is declared unconditionally above (:259) — this
        // branch only adds the cfg itself. (A second rustc-check-cfg here was
        // a duplicate, removed per AUDIT-2026-09-07 ffi-4.)
        println!("cargo:rustc-cfg=has_qtsq");
        // Rebuild when the checkout's artifacts change — without these a
        // refreshed libqtsq.a silently keeps the stale probe cfg and (until
        // some downstream crate edits) the stale link inputs.
        println!("cargo:rerun-if-changed={}", qtsq_lib.display());
        println!("cargo:rerun-if-changed={}", qtsq_dir.join("include/qtsq_format.h").display());
        println!("cargo:rerun-if-changed={}", qtsq_dir.join("include/qtsq.h").display());
    } else if qtsq_dir.is_some() {
        println!(
            "cargo:warning=QTSQ static lib not found at {} — building WITHOUT QTSQ \
             (session store + brain persistence disabled). Point SOFUU_QTSQ_DIR \
             at a built QTSQ checkout (SOFUU_QTSQ_LIB overrides the lib path).",
            qtsq_lib.display()
        );
    } else {
        println!(
            "cargo:warning=SOFUU_QTSQ_DIR not set — building WITHOUT QTSQ \
             (session store + brain persistence disabled). Set it to a built \
             QTSQ checkout to enable them."
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
    println!("cargo:rerun-if-env-changed=SOFUU_QTSQ_LIB");
    println!("cargo:rerun-if-env-changed=SOFUU_ZLIB_DIR");
    // ffi-4 (AUDIT-2026-09-07): the libuv/curl probes read these too — without
    // the declarations, flipping them (e.g. dist scripts switching uv/curl
    // source dirs) would NOT re-run this build script and the binary kept
    // linking the previous libraries.
    println!("cargo:rerun-if-env-changed=SOFUU_UV_DIR");
    println!("cargo:rerun-if-env-changed=SOFUU_CURL_STATIC_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_INSTALLATION_ROOT");

    // ── libopus (optional, dev/test links) ─────────────────────────
    // The POSIX QTSQ archive's audio objects reference libopus. Release
    // links (LTO) prune those objects, but debug/test links pull them →
    // link fails without it. Add it when pkg-config finds it — only in
    // QTSQ builds. P2-21 (AUDIT-2026-09-01): also gate on HOST == TARGET —
    // a HOST pkg-config result poisoned CROSS links before (the flag was
    // emitted from the build machine's opus, wrong arch). Windows: the
    // CMake port excludes qtsq_audio_opus.c entirely (audio falls back to
    // the in-house spectral encoder, same as a POSIX build without
    // libopus), so there's nothing to link — and no pkg-config.
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    if has_qtsq && !is_windows && host == target {
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
    // qtsq_format.h via the cc crate (finds cc/clang on POSIX, cl.exe on
    // Windows — a raw `cc` spawn has no MSVC equivalent). The object lands
    // in OUT_DIR and nothing is executed. A checkout that changes
    // sizeof/offsets fails the BUILD here instead of silently corrupting
    // brain files at runtime. Bump these expected values together with
    // the Rust mirrors in crates/sofuu-ffi/src/qtsq.rs.
    if has_qtsq {
        let qtsq_dir = qtsq_dir.as_deref().expect("has_qtsq implies SOFUU_QTSQ_DIR");
        let guard_src = out_dir.join("qtsq_layout_guard.c");
        std::fs::write(
            &guard_src,
            r#"#include <stddef.h>
#include "qtsq_format.h"
/* cc's default archive step warns about an empty TOC on macOS — a single
 * INITIALIZED symbol keeps the object in the .a as a defined global
 * (an uninitialized one becomes a common symbol, which ranlib ignores)
 * without affecting the asserts. */
int qtsq_layout_guard_present = 1;
_Static_assert(sizeof(qtsq_context_t) == 5024, "QTSQ_CONTEXT_SIZE drifted");
_Static_assert(offsetof(qtsq_context_t, header.data_type) == 10, "OFF_HEADER_DATA_TYPE drifted");
_Static_assert(offsetof(qtsq_context_t, schema) == 112, "OFF_SCHEMA drifted");
_Static_assert(offsetof(qtsq_schema_t, dimensions) == 68, "OFF_SCHEMA_DIMENSIONS drifted");
_Static_assert(offsetof(qtsq_schema_t, num_dims) == 100, "OFF_SCHEMA_NUM_DIMS drifted");
_Static_assert(offsetof(qtsq_context_t, is_encrypted) == 4908, "OFF_IS_ENCRYPTED drifted");
_Static_assert(offsetof(qtsq_context_t, codec) == 5016, "OFF_CODEC drifted");
_Static_assert(offsetof(qtsq_context_t, quality) == 5020, "OFF_QUALITY drifted");
"#,
        )
        .expect("write qtsq layout guard");
        let compiled = cc::Build::new()
            .file(&guard_src)
            .include(qtsq_dir.join("include"))
            .std("c11")
            .warnings(false)
            .try_compile("qtsq_layout_guard");
        if let Err(e) = compiled {
            panic!(
                "QTSQ layout guard failed: the checkout's qtsq_context_t no \
                 longer matches the Rust mirrors in qtsq.rs (sizeof=5024, \
                 offsets 10/112/68/100/4908). Update the mirrors and this \
                 guard together.\n{e}"
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", src.display());
    // The vendored C trees live outside src/ — without these a QuickJS or
    // libuv edit keeps the last build script output (and stale .a) silently.
    println!("cargo:rerun-if-changed={}", deps.join("quickjs").display());
    println!("cargo:rerun-if-changed={}", deps.join("libuv/include").display());
    println!("cargo:rerun-if-changed={}", deps.join("http-parser").display());
}
