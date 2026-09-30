#!/usr/bin/env bash
# scripts/cross/build_windows_x86_64.sh
#
# Cross-compile Sofuu for Windows x64 from macOS (or CI) using Zig as CC.
# Zig bundles the mingw-w64 libc, CRT, and import libs — no Visual Studio,
# no vcpkg, no Docker needed. (Native Windows builds use the MSVC recipe in
# sofuu-desktop/README.md; this script is the no-Docker cross path.)
#
# M10: the runtime is Rust (crates/) over the C engine layer, so this runs
# a cargo cross-build: rustc links with zig (mingw target), the cc crate
# compiles QuickJS/SIMD/http-parser with `zig cc` (via zig-cc.sh), and
# libuv + libcurl are prebuilt for the target (dist/). libcurl is built
# WITH Schannel TLS — Windows's own TLS provider, so https works natively.
# QTSQ is skipped — the proprietary checkout links from the local checkout
# only; cross builds degrade exactly like CI (no session persistence).
#
# Usage:
#   bash scripts/cross/install_zig.sh     # once
#   bash scripts/cross/build_windows_x86_64.sh
#
# Output: dist/sofuu-windows-x86_64.exe
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

TARGET="x86_64-windows-gnu"
RUST_TARGET="x86_64-pc-windows-gnu"
OUT="$DIST/sofuu-windows-x86_64.exe"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cross-compiling Sofuu → Windows x64 (mingw static)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$DIST"
cd "$REPO_ROOT"

# ── rustup target (std for windows-gnu) ─────────────────────────
if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
    echo "→ Adding rustup target ${RUST_TARGET}..."
    rustup target add "$RUST_TARGET"
fi

# ── Build libuv for the target ──────────────────────────────────
echo "→ Building libuv for ${TARGET}..."
UV_DIR="deps/libuv"
UV_BUILD="$UV_DIR/build-win-x64"

# Compiler/AR/RANLIB via baked-target wrappers: cmake's CMAKE_C_COMPILER
# must be a single executable (ARG1 is unreliable), and with
# CMAKE_SYSTEM_NAME=Windows cmake groups objects into objects.a via
# CMAKE_AR — the host /usr/bin/ar can't read the @response-file args it
# passes there, so ar must be zig's.
rm -rf "$UV_BUILD"
cmake -S "$UV_DIR" -B "$UV_BUILD" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER="$REPO_ROOT/scripts/cross/zig-cc-win.sh" \
    -DCMAKE_AR="$REPO_ROOT/scripts/cross/zig-ar.sh" \
    -DCMAKE_RANLIB="$REPO_ROOT/scripts/cross/zig-ranlib.sh" \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    2>&1 | tail -3
cmake --build "$UV_BUILD" --target uv_a -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5

UV_LIB="$(ls "$UV_BUILD"/libuv*.a | head -1)"
echo "✓ libuv built: $UV_LIB"

# ── Build libcurl for the target (Schannel TLS) ─────────────────
CURL_VER="8.6.0"
CURL_SRC="/tmp/curl-${CURL_VER}"
CURL_TARBALL="/tmp/curl-${CURL_VER}.tar.gz"
# P1-17 (AUDIT-2026-09-07): pin the tarball hash — same 8.6.0 artifact as
# build_libcurl_static.sh's P2-24 pin (verified against the curl.se PGP
# signature) — and gate /tmp source reuse on a marker naming that exact
# hash, so a stale (pre-pin) or tampered /tmp tree is never built.
CURL_SHA256="9c6db808160015f30f3c656c0dec125feb9dc00753596bf858a272b5dd8dc398"
CURL_MARK="${CURL_SRC}/.verified-sha"
CURL_INSTALL="$DIST/libcurl-$TARGET"
CURL_LIB="$CURL_INSTALL/lib"
CURL_BUILD="${CURL_SRC}/build-${TARGET}"

if [ -f "$CURL_LIB/libcurl.a" ]; then
    echo "✓ libcurl already built for $TARGET"
else
    if [ ! -f "$CURL_TARBALL" ]; then
        echo "→ Downloading curl ${CURL_VER}..."
        curl -fsSL "https://curl.se/download/curl-${CURL_VER}.tar.gz" -o "$CURL_TARBALL"
    fi
    GOT_SHA="$(shasum -a 256 "$CURL_TARBALL" | cut -d' ' -f1)"
    if [ "$GOT_SHA" != "$CURL_SHA256" ]; then
        echo "✗ curl tarball checksum mismatch:"
        echo "    expected $CURL_SHA256"
        echo "    got      $GOT_SHA"
        echo "  Refusing to build from an unverified download."
        exit 1
    fi
    # Reuse the extracted tree only when it provably came from the verified
    # tarball; anything else is wiped and re-extracted from it.
    if [ ! -d "$CURL_SRC" ] || [ ! -f "$CURL_MARK" ] || [ "$(cat "$CURL_MARK")" != "$CURL_SHA256" ]; then
        rm -rf "$CURL_SRC"
        tar -xf "$CURL_TARBALL" -C /tmp
        printf '%s\n' "$CURL_SHA256" > "$CURL_MARK"
    fi
    echo "→ Configuring curl for ${TARGET} (Schannel TLS)..."
    mkdir -p "$CURL_BUILD" "$CURL_LIB"
    rm -rf "$CURL_BUILD"
    cmake -S "$CURL_SRC" -B "$CURL_BUILD" \
        -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_INSTALL_PREFIX="$CURL_INSTALL" \
        -DCMAKE_C_COMPILER="$REPO_ROOT/scripts/cross/zig-cc-win.sh" \
        -DCMAKE_AR="$REPO_ROOT/scripts/cross/zig-ar.sh" \
        -DCMAKE_RANLIB="$REPO_ROOT/scripts/cross/zig-ranlib.sh" \
        -DCMAKE_EXE_LINKER_FLAGS="-static" \
        -DCMAKE_SYSTEM_NAME=Windows \
        -DCMAKE_SYSTEM_PROCESSOR=AMD64 \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_TESTING=OFF \
        -DCURL_USE_SCHANNEL=ON \
        -DCURL_DISABLE_LDAP=ON \
        -DCURL_DISABLE_LDAPS=ON \
        -DCURL_DISABLE_RTSP=ON \
        -DCURL_DISABLE_POP3=ON \
        -DCURL_DISABLE_IMAP=ON \
        -DCURL_DISABLE_SMTP=ON \
        -DCURL_DISABLE_GOPHER=ON \
        -DCURL_DISABLE_FTP=ON \
        -DCURL_DISABLE_TELNET=ON \
        -DCURL_DISABLE_TFTP=ON \
        -DCURL_ZLIB=OFF \
        -DUSE_NGHTTP2=OFF \
        -DENABLE_ARES=OFF \
        2>&1 | tail -5
    echo "→ Building curl..."
    cmake --build "$CURL_BUILD" -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5
    cmake --install "$CURL_BUILD"
fi

# ── Cargo cross-build (compiles QuickJS/SIMD/http-parser via zig cc) ──
echo "→ cargo build --release --target ${RUST_TARGET} -p sofuu-core"
echo "   (cc crate → zig cc via zig-cc.sh wrapper; linker → zig)"
# cc-rs injects the rust triple as --target=... — the zig-cc.sh wrapper
# rewrites it to the zig spelling (see zig-cc.sh).
export CC="$REPO_ROOT/scripts/cross/zig-cc.sh"
export CXX="$REPO_ROOT/scripts/cross/zig-cxx.sh --cxx"
export AR="$ZIG ar"
export RANLIB="$ZIG ranlib"
# Absolute: cargo runs build.rs with cwd = crates/sofuu-ffi, so a relative
# path here resolves against the crate dir and the probe misses the archive.
export SOFUU_UV_DIR="$REPO_ROOT/$UV_BUILD"
export SOFUU_CURL_STATIC_DIR="$CURL_LIB"
# QTSQ persistence: the zig-built single-archive port (qtsq.lib, qtc +
# POSIX shim baked in — mirrors windows/CMakeLists.txt) from
# scripts/cross/build_qtsq_cross.sh (run it first). Fallback: an empty dir
# keeps the build alive as a degraded no-QTSQ binary.
# SOFUU_QTSQ_DIR must contain libqtsq.a itself: sofuu-core's build.rs
# probes that path for its has_qtsq cfg (the memory shells), while
# sofuu-ffi links SOFUU_QTSQ_LIB. Same dir covers both.
export SOFUU_QTSQ_DIR="$DIST/qtsq-windows-x86_64"
export SOFUU_QTSQ_LIB="$DIST/qtsq-windows-x86_64/qtsq.lib"
export SOFUU_ZLIB_DIR="$DIST/qtsq-windows-x86_64"
if [ ! -f "$SOFUU_QTSQ_LIB" ]; then
    echo "WARNING: dist/qtsq-windows-x86_64/qtsq.lib missing — building WITHOUT QTSQ." >&2
    echo "  Run scripts/cross/build_qtsq_cross.sh first." >&2
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"
    unset SOFUU_QTSQ_LIB SOFUU_ZLIB_DIR
fi
# link-self-contained=no: the rust-mingw component is not installed — let
# zig's bundled mingw-w64 CRT/import libs provide everything instead.
# -C linker via zig-ld.sh: rustc emits its own leading flags (-m64) before
# any link-arg, so a bare `zig`/`link-arg=cc` chain can't work — the wrapper
# guarantees `zig cc` is argv[0..1].
export RUSTFLAGS="-C linker=$REPO_ROOT/scripts/cross/zig-ld.sh -C link-arg=-target -C link-arg=$TARGET -C link-self-contained=no"

# -p sofuu-core: build only the CLI package (see build_linux_x86_64.sh —
# the desktop app's notification stack needs target libdbus/pkg-config).
cargo build --release --target "$RUST_TARGET" -p sofuu-core 2>&1 | tail -20

cp "target/${RUST_TARGET}/release/sofuu.exe" "$OUT"
SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Built: $OUT ($SIZE)"
echo "   Target: $TARGET (mingw static, runs on any Windows x64 8.1+)"
echo "   TLS: Schannel (Windows native)   QTSQ: $([ -n "${SOFUU_QTSQ_LIB:-}" ] && echo linked || echo off)"
echo ""
