#!/usr/bin/env bash
# scripts/cross/build_linux_arm64.sh
#
# Cross-compile Sofuu for Linux arm64 (aarch64) from macOS using Zig as CC.
#
# M10: the runtime is Rust (crates/) over the C engine layer, so this now
# runs a cargo cross-build: rustc links with zig (static musl), the cc
# crate compiles QuickJS/SIMD/http-parser with `zig cc`, and libuv +
# libcurl are prebuilt for the target (dist/). QTSQ is skipped — the
# proprietary checkout is macOS-only; CI covers the same degraded build.
#
# Usage:
#   bash scripts/cross/install_zig.sh     # once
#   bash scripts/cross/build_linux_arm64.sh
#
# Output: dist/sofuu-linux-arm64  (statically linked)
#
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"

if [ ! -x "$ZIG" ]; then
    echo "Zig not found. Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

TARGET="aarch64-linux-musl"
RUST_TARGET="aarch64-unknown-linux-musl"
OUT="$DIST/sofuu-linux-arm64"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Linux arm64 (musl static)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── rustup target (std for musl) ─────────────────────────────────
if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
    echo "→ Adding rustup target ${RUST_TARGET}..."
    rustup target add "$RUST_TARGET"
fi

# ── Build libuv for aarch64-linux-musl ─────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-linux-arm64"

cmake -S "$UV_DIR" -B "$UV_BUILD" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER="$ZIG" \
    -DCMAKE_C_COMPILER_ARG1="cc -target $TARGET" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
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
# scripts/cross/build_qtsq_cross.sh (run it first). Fallback: an empty dir
# keeps the build alive as a degraded no-QTSQ binary.
export SOFUU_QTSQ_DIR="$DIST/qtsq-linux-arm64"
export SOFUU_ZLIB_DIR="$DIST/qtsq-linux-arm64"
if [ ! -f "$SOFUU_QTSQ_DIR/libqtsq.a" ]; then
    echo "WARNING: dist/qtsq-linux-arm64/libqtsq.a missing — building WITHOUT QTSQ." >&2
    echo "  Run scripts/cross/build_qtsq_cross.sh first." >&2
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"
    unset SOFUU_ZLIB_DIR
fi
# -C linker via zig-ld.sh: rustc emits its own leading flags (-m64) before
# any link-arg, so a bare `zig`/`link-arg=cc` chain can't work — the wrapper
# guarantees `zig cc` is argv[0..1].
export RUSTFLAGS="-C linker=$REPO_ROOT/scripts/cross/zig-ld.sh -C link-arg=-target -C link-arg=$TARGET -C link-self-contained=no"
# link-self-contained=no: rustc's bundled musl CRT objects (rcrt1.o/crti.o)
# duplicate the ones zig's driver links itself → duplicate-symbol errors.
# With it off, zig supplies the CRT for the -static -pie link.

# -p sofuu-core: build only the CLI package. A whole-workspace build drags
# in the desktop app (sofuu-desktop/src-tauri), whose notification stack
# (tauri-plugin-notification → notify-rust → libdbus-sys) requires a
# target pkg-config for libdbus — unusable on a cross host.
cargo build --release --target "$RUST_TARGET" -p sofuu-core 2>&1 | tail -20

cp "target/${RUST_TARGET}/release/sofuu" "$OUT"
chmod +x "$OUT"

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: $TARGET (static musl, runs on Graviton / RPi / Oracle ARM)"
echo "   QTSQ: $([ -f "$SOFUU_QTSQ_DIR/libqtsq.a" ] && echo linked || echo OFF)"
echo ""
