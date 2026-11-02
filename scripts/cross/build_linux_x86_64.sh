#!/usr/bin/env bash
# scripts/cross/build_linux_x86_64.sh
#
# Cross-compile Sofuu for Linux x86_64 from macOS (or CI) using Zig as CC.
# Zig bundles its own libc and linker — no sysroot or musl package needed.
#
# M10: the runtime is Rust (crates/) over the C engine layer, so this now
# runs a cargo cross-build: rustc links with zig (static musl), the cc
# crate compiles QuickJS/SIMD/http-parser with `zig cc`, and libuv +
# libcurl are prebuilt for the target (dist/). QTSQ is a mandatory core
# component (sofuu-ffi/build.rs refuses to configure without it), so it must be
# cross-compiled first with scripts/cross/build_qtsq_cross.sh.
#
# Usage:
#   bash scripts/cross/install_zig.sh     # once
#   bash scripts/cross/build_linux_x86_64.sh
#
# Output: dist/sofuu-linux-x86_64  (statically linked)
#
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"

if [ ! -x "$ZIG" ]; then
    echo "Zig not found at $ZIG"
    echo "Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

TARGET="x86_64-linux-musl"
RUST_TARGET="x86_64-unknown-linux-musl"
OUT="$DIST/sofuu-linux-x86_64"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Linux x86_64 (musl static)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── rustup target (std for musl) ─────────────────────────────────
if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
    echo "→ Adding rustup target ${RUST_TARGET}..."
    rustup target add "$RUST_TARGET"
fi

# ── Build libuv for the target ──────────────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-linux-x86_64"

# Use zig cc as CMake toolchain via environment variables
export CC="$ZIG cc -target $TARGET"
export CXX="$ZIG c++ -target $TARGET"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"

cmake -S "$UV_DIR" -B "$UV_BUILD" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER="$ZIG" \
    -DCMAKE_C_COMPILER_ARG1="cc -target $TARGET" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
    -DCMAKE_C_FLAGS="-target $TARGET" \
    -DCMAKE_EXE_LINKER_FLAGS="-target $TARGET -static" \
    2>&1 | tail -3
cmake --build "$UV_BUILD" --target uv_a -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5

UV_LIB="$UV_BUILD/libuv.a"
echo "✓ libuv built: $UV_LIB"

# ── Build libcurl for the target ────────────────────────────────
echo "→ Building static libcurl for ${TARGET}..."
bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$TARGET"
CURL_INSTALL="$DIST/libcurl-$TARGET"
CURL_LIB="$CURL_INSTALL/lib"

# ── Cargo cross-build (compiles QuickJS/SIMD/http-parser via zig cc) ──
echo "→ cargo build --release --target ${RUST_TARGET} -p sofuu-core"
echo "   (cc crate → zig cc via zig-cc.sh wrapper; linker → zig)"
# cc-rs injects the rust triple as --target=... — the zig-cc.sh wrapper
# rewrites it to the zig spelling (pre-seeding -target here is not enough,
# the last --target wins and zig rejects the rust spelling).
export CC="$REPO_ROOT/scripts/cross/zig-cc.sh"
export CXX="$REPO_ROOT/scripts/cross/zig-cxx.sh --cxx"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"
# Absolute: cargo runs build.rs with cwd = crates/sofuu-ffi, so a relative
# path here resolves against the crate dir and the probe misses the archive.
export SOFUU_UV_DIR="$REPO_ROOT/$UV_BUILD"
export SOFUU_CURL_STATIC_DIR="$CURL_LIB"
# QTSQ persistence: link the QTSQ static libs cross-compiled by
# scripts/cross/build_qtsq_cross.sh (run it first).
#
# There is no degraded no-QTSQ build to fall back to: sofuu-ffi/build.rs treats
# QTSQ as a mandatory core component and refuses to configure without it, so an
# empty SOFUU_QTSQ_DIR does not keep the build alive — it only moves the failure
# deeper, into a build-script panic. Fail here, where the fix is obvious.
export SOFUU_QTSQ_DIR="$DIST/qtsq-linux-x86_64"
export SOFUU_ZLIB_DIR="$DIST/qtsq-linux-x86_64"
if [ ! -f "$SOFUU_QTSQ_DIR/libqtsq.a" ]; then
    echo "ERROR: $SOFUU_QTSQ_DIR/libqtsq.a is missing, and QTSQ cannot be skipped." >&2
    echo "  Run: bash scripts/cross/build_qtsq_cross.sh" >&2
    exit 1
fi
# -C linker via zig-ld.sh: rustc emits its own leading flags (-m64) before
# any link-arg, so a bare `zig`/`link-arg=cc` chain can't work — the wrapper
# guarantees `zig cc` is argv[0..1].
# PC-07 (AUDIT-2026-10-29): the same --remap-path-prefix set `make` uses, so
# the published binary carries no build-host absolute path. Textual prefix
# substitution — it does not touch the linker or the cc invocation.
REMAP="$(bash "$REPO_ROOT/scripts/cross/remap_flags.sh")"
export RUSTFLAGS="$REMAP -C linker=$REPO_ROOT/scripts/cross/zig-ld.sh -C link-arg=-target -C link-arg=$TARGET -C link-self-contained=no"
# link-self-contained=no: rustc's bundled musl CRT objects (rcrt1.o/crti.o)
# duplicate the ones zig's driver links itself → duplicate-symbol errors.
# With it off, zig supplies the CRT for the -static -pie link.

# ── Lean link (2026-11-02) ──────────────────────────────────────
# Measured on this tree: Linux x86_64 was 7,735,896 B against a 5,767,168 B
# cap, and a third of the overage was link furniture, not code. A static-PIE
# musl link pays for two things a static non-PIE binary does not need:
#   -C relocation-model=static + -no-pie   → .rela.dyn 380,064 B → 0,
#                                            .data.rel.ro 268,888 B → 47,960 B
#   -C force-unwind-tables=no              → no unwind tables from Rust
#   -Wl,--no-eh-frame-hdr                  → no ELF header for a .eh_frame
# Result: 7,316,648 B, i.e. 419,248 B smaller. Still over the cap — the rest is
# C-side .eh_frame (667 KB) and .text, tracked in
# docs/PLAN-AUDIT-REMEDIATION-2026-11-02.md §9 Task 7.1.
# SOFUU_LEAN_BUILD=0 falls back to the old picture-based link if a future musl
# or zig release objects; a lean artifact must be smoke-tested on Linux before
# it is published, because none of this changes the size gate's own checks.
if [ "${SOFUU_LEAN_BUILD:-1}" != "0" ]; then
    RUSTFLAGS="$RUSTFLAGS -C force-unwind-tables=no -C relocation-model=static"
    RUSTFLAGS="$RUSTFLAGS -C link-arg=-no-pie -C link-arg=-Wl,--no-eh-frame-hdr"
    # The C dependencies (QuickJS, http-parser) ship unwind tables by default
    # and nothing C here throws: worth 77,216 B on a clean tree. Must be set
    # before cargo runs, and it only takes effect when the C objects are
    # rebuilt — cc-rs caches them, so measure it with a clean target dir or the
    # byte count will lie.
    NOUNWIND="-fno-asynchronous-unwind-tables -fno-unwind-tables"
    export CFLAGS="${CFLAGS:-} $NOUNWIND"
    export CFLAGS_x86_64_unknown_linux_musl="${CFLAGS_x86_64_unknown_linux_musl:-} $NOUNWIND"
    export CFLAGS_arm64_unknown_linux_musl="${CFLAGS_arm64_unknown_linux_musl:-} $NOUNWIND"
    export CXXFLAGS="${CXXFLAGS:-} $NOUNWIND"
fi
export RUSTFLAGS

# ── Build std with the panic machinery removed (SOFUU_BUILD_STD) ──────
#
# Worth 1,151,672 B: 7,239,432 → 6,087,760, measured on a clean tree with the
# flags above already in place. panic = "immediate-abort" is a real panic
# strategy as of the nightly used here, and it lets the compiler drop the
# formatting machinery behind every panic message plus std's own unwind tables
# (.eh_frame fell from 599,664 B to 138,792 B). The release profile already says
# panic = "abort", so nothing about the program's behaviour changes: a panic was
# already an abort, this only stops compiling the code that explains it.
#
# It needs a NIGHTLY toolchain with rust-src:
#     rustup toolchain install nightly
#     rustup component add rust-src --toolchain nightly
# SOFUU_BUILD_STD=0 keeps the stable build, which is 1.15 MB larger and over the
# 5,767,168 B cap in RULES.md §3.1.
BUILD_STD="${SOFUU_BUILD_STD:-1}"
CARGO_CMD=(cargo build)
if [ "$BUILD_STD" != "0" ]; then
    if ! rustc +nightly --version >/dev/null 2>&1; then
        echo "ERROR: SOFUU_BUILD_STD=1 needs a nightly toolchain (rustup toolchain install nightly)." >&2
        echo "       Set SOFUU_BUILD_STD=0 to build with stable, 1.15 MB over the cap." >&2
        exit 1
    fi
    if [ ! -d "$(rustc +nightly --print sysroot)/lib/rustlib/src/rust/library" ]; then
        echo "ERROR: nightly is missing rust-src (rustup component add rust-src --toolchain nightly)." >&2
        exit 1
    fi
    RUSTFLAGS="$RUSTFLAGS -Zunstable-options -Cpanic=immediate-abort -Zlocation-detail=none -Zfmt-debug=none"
    CARGO_CMD=(cargo +nightly build -Z build-std=std,panic_abort)
    echo "→ building std from source with panic=immediate-abort"
fi
export RUSTFLAGS

# -p sofuu-core: build only the CLI package. A whole-workspace build drags
# in the desktop app (sofuu-desktop/src-tauri), whose notification stack
# (tauri-plugin-notification → notify-rust → libdbus-sys) requires a
# target pkg-config for libdbus — unusable on a cross host.
"${CARGO_CMD[@]}" --release --target "$RUST_TARGET" -p sofuu-core 2>&1 | tail -20

cp "target/${RUST_TARGET}/release/sofuu" "$OUT"
chmod +x "$OUT"

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: $TARGET (static musl, runs on any Linux x86_64)"
echo "   QTSQ: $([ -f "$SOFUU_QTSQ_DIR/libqtsq.a" ] && echo linked || echo OFF)"
echo ""
