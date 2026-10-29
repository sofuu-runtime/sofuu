#!/usr/bin/env bash
# build_linux_capi_so.sh — build the embeddable C ABI as a LINUX SHARED
# LIBRARY (libsofuu.so) and stage it in dist/.
#
# WHY glibc AND NOT musl:
#   rustc DROPS the `cdylib` crate type for every *-linux-musl target — the
#   target spec has crt_static_allows_dylibs=false — so
#       cargo build --target x86_64-unknown-linux-musl -p sofuu-capi
#   prints "dropping unsupported crate type `cdylib` for target
#   `x86_64-unknown-linux-musl`" and emits ONLY libsofuu_capi.a. A musl
#   libsofuu.so cannot exist; asking for one is why the C-ABI contract rows
#   stayed blocked. The CLI binary keeps its static-musl build — only the
#   embeddable .so is a glibc (DT_NEEDED libc.so.6) artifact.
#
# Usage:
#   bash scripts/cross/build_linux_capi_so.sh [x86_64]
#
# Output:
#   dist/libsofuu-linux-x86_64.so   platform-pack name
#   dist/libsofuu.so                the name scripts/contract and
#                                   scripts/check_abi.sh auto-detect
#   dist/sofuu_embed.h              the matching header
#
# Prerequisites: Rust + the x86_64-unknown-linux-gnu target, Zig
# (scripts/cross/install_zig.sh), cmake, and the QTSQ checkout at
# ~/projects/black-hole-disk (build_qtsq_cross.sh compiles it; QTSQ is
# mandatory, sofuu-ffi/build.rs panics without it).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"
ZIG_TARGET="x86_64-linux-gnu"
RUST_TARGET="x86_64-unknown-linux-gnu"
ARCH="${1:-x86_64}"
if [ "$ARCH" != "x86_64" ]; then
    echo "Usage: $0 [x86_64]   (only x86_64 is wired today)" >&2
    exit 1
fi

cd "$REPO_ROOT"
[ -x "$ZIG" ] || { echo "Zig not found at $ZIG — run scripts/cross/install_zig.sh" >&2; exit 1; }
mkdir -p "$DIST"
if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
    rustup target add "$RUST_TARGET"
fi

# ── libuv: PIC static archive for the glibc target ──────────────────
UV_BUILD="$REPO_ROOT/deps/libuv/build-linux-gnu-x86_64"
if [ ! -f "$UV_BUILD/libuv.a" ]; then
    echo "→ libuv (PIC) for $ZIG_TARGET"
    export AR="$ZIG ar" RANLIB="$ZIG ranlib"
    cmake -S deps/libuv -B "$UV_BUILD" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_C_COMPILER="$ZIG" \
        -DCMAKE_C_COMPILER_ARG1="cc -target $ZIG_TARGET" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
        -DCMAKE_C_FLAGS="-target $ZIG_TARGET -fPIC" 2>&1 | tail -2
    cmake --build "$UV_BUILD" --target uv_a -- -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" 2>&1 | tail -2
fi

# ── QTSQ (mandatory) + static libcurl, both for the glibc target ────
bash "$REPO_ROOT/scripts/cross/build_qtsq_cross.sh" >/dev/null
QTSQ_DIR="$DIST/qtsq-linux-x86_64-gnu"
[ -f "$QTSQ_DIR/libqtsq.a" ] || { echo "✗ QTSQ not built for glibc ($QTSQ_DIR)" >&2; exit 1; }
bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$ZIG_TARGET" >/dev/null
CURL_LIB="$DIST/libcurl-$ZIG_TARGET/lib"

# ── cargo → CDYLIB ─────────────────────────────────────────────────
# panic=unwind: the embedded ABI contains panics at the boundary
# (SOFUU_ERR_PANIC) instead of aborting the host process — same override
# `make libsofuu` applies to the capi graph.
# -Wl,-soname: rustc emits no DT_SONAME for a Linux cdylib, so a host that
# links -lsofuu would bake the build-tree path into its DT_NEEDED.
echo "→ cargo build --release --target $RUST_TARGET -p sofuu-capi"
export CARGO_PROFILE_RELEASE_PANIC=unwind
export CC="$REPO_ROOT/scripts/cross/zig-cc.sh"
export CXX="$REPO_ROOT/scripts/cross/zig-cxx.sh --cxx"
export AR="$ZIG ar" RANLIB="$ZIG ranlib"
export SOFUU_UV_DIR="$UV_BUILD"
export SOFUU_CURL_STATIC_DIR="$CURL_LIB"
export SOFUU_QTSQ_DIR="$QTSQ_DIR"
export SOFUU_ZLIB_DIR="$QTSQ_DIR"
export RUSTFLAGS="$(bash "$REPO_ROOT/scripts/cross/remap_flags.sh") \
 -C linker=$REPO_ROOT/scripts/cross/zig-ld.sh \
 -C link-arg=-target -C link-arg=$ZIG_TARGET \
 -C link-arg=-Wl,-soname=libsofuu.so"
cargo build --release --target "$RUST_TARGET" -p sofuu-capi

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
SO="$TARGET_DIR/$RUST_TARGET/release/libsofuu_capi.so"
[ -f "$SO" ] || SO="$TARGET_DIR/$RUST_TARGET/release/libsofuu.so"
[ -f "$SO" ] || { echo "✗ no cdylib produced for $RUST_TARGET" >&2; exit 1; }
cp "$SO" "$DIST/libsofuu-linux-x86_64.so"
cp "$SO" "$DIST/libsofuu.so"
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h"

echo ""
echo "✅ Built: $DIST/libsofuu-linux-x86_64.so ($(du -h "$DIST/libsofuu-linux-x86_64.so" | cut -f1))"
echo "   also staged as dist/libsofuu.so (contract-suite auto-detect name)"
echo ""
echo "   Verify on the Linux host (no binutils needed — the loader is the check):"
echo "     python3 -c \"import ctypes;ctypes.CDLL('./dist/libsofuu.so').sofuu_rt_call\" && echo OK"
echo "   Verify the exports here:  bash scripts/check_abi.sh dist/libsofuu-linux-x86_64.so"
echo ""
