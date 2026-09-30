#!/usr/bin/env bash
# scripts/dist/linux.sh — H4 platform pack: Linux (x86_64 + arm64, static musl)
#
# Produces: dist/libsofuu-linux-{x86_64,arm64}.a
#           dist/libsofuu-linux-{x86_64,arm64}.so
#           dist/sofuu_embed.h
#           dist/libsofuu-linux-{x86_64,arm64}.tar.gz
#
# Usage: bash scripts/dist/linux.sh [arch]
#   arch: x86_64 | arm64 | all (default: all)
#
# Prerequisites: Zig (scripts/cross/install_zig.sh), Rust toolchain, cmake.
#
# This script reuses the cross-compile infrastructure from scripts/cross/
# but targets the capi crate (libsofuu) instead of the CLI binary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
ZIG="$REPO_ROOT/tools/zig/zig"
ARCH_ARG="${1:-all}"

if [ ! -x "$ZIG" ]; then
    echo "  Zig not found at: $ZIG"
    echo "  Install with: bash scripts/cross/install_zig.sh"
    exit 1
fi

build_arch() {
    local ARCH="$1"
    local ZIG_TARGET=""
    local RUST_TARGET=""
    case "$ARCH" in
        x86_64) ZIG_TARGET="x86_64-linux-musl";  RUST_TARGET="x86_64-unknown-linux-musl" ;;
        arm64)   ZIG_TARGET="aarch64-linux-musl"; RUST_TARGET="aarch64-unknown-linux-musl" ;;
        *) echo "Unknown arch: $ARCH"; exit 1 ;;
    esac

    echo ""
    echo "  \033[36mBuilding libsofuu for Linux $ARCH ($RUST_TARGET, static musl)\033[0m"

    cd "$REPO_ROOT"

    # Ensure the rustup target is installed.
    if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
        rustup target add "$RUST_TARGET"
    fi

    # Build libuv for the target.
    local UV_DIR="deps/libuv"
    local UV_BUILD="deps/libuv/build-linux-$ARCH"
    export CC="$ZIG cc -target $ZIG_TARGET"
    export CXX="$ZIG c++ -target $ZIG_TARGET"
    export AR="$ZIG ar"
    export RANLIB="$ZIG ranlib"

    cmake -S "$UV_DIR" -B "$UV_BUILD" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_C_COMPILER="$ZIG" \
        -DCMAKE_C_COMPILER_ARG1="cc -target $ZIG_TARGET" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR="$ARCH" \
        -DCMAKE_C_FLAGS="-target $ZIG_TARGET -fPIC" \
        2>&1 | tail -3
    cmake --build "$UV_BUILD" --target uv_a 2>&1 | tail -3

    # Build libcurl for the target.
    bash "$REPO_ROOT/scripts/cross/build_libcurl_static.sh" "$ZIG_TARGET"
    local CURL_INSTALL="$DIST/libcurl-$ZIG_TARGET"

    export SOFUU_UV_DIR="$UV_BUILD"
    export SOFUU_CURL_STATIC_DIR="$CURL_INSTALL/lib"
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"
    export RUSTFLAGS="-C linker=$ZIG -C link-arg=cc -C link-arg=-target -C link-arg=$ZIG_TARGET"

    cargo build --release -p sofuu-capi --target "$RUST_TARGET"

    # Copy artifacts.
    mkdir -p "$DIST"
    local LIB_DIR="target/${RUST_TARGET}/release"
    cp "$LIB_DIR/libsofuu_capi.a" "$DIST/libsofuu-linux-$ARCH.a" 2>/dev/null || \
        cp "$LIB_DIR/libsofuu.a" "$DIST/libsofuu-linux-$ARCH.a" 2>/dev/null || true
    cp "$LIB_DIR/libsofuu_capi.so" "$DIST/libsofuu-linux-$ARCH.so" 2>/dev/null || \
        cp "$LIB_DIR/libsofuu.so" "$DIST/libsofuu-linux-$ARCH.so" 2>/dev/null || true

    # Create tarball with header.
    local TARBALL_DIR="$DIST/libsofuu-linux-$ARCH"
    mkdir -p "$TARBALL_DIR"
    cp "$DIST/libsofuu-linux-$ARCH.a" "$TARBALL_DIR/" 2>/dev/null || true
    cp "$DIST/libsofuu-linux-$ARCH.so" "$TARBALL_DIR/" 2>/dev/null || true
    cp include/sofuu_embed.h "$TARBALL_DIR/"
    tar -czf "$DIST/libsofuu-linux-$ARCH.tar.gz" -C "$DIST" "$(basename $TARBALL_DIR)"
    rm -rf "$TARBALL_DIR"

    local SIZE=$(du -sh "$DIST/libsofuu-linux-$ARCH.a" 2>/dev/null | cut -f1)
    echo "  \033[32m✓\033[0m libsofuu-linux-$ARCH.a ($SIZE)"

    # Size check (5MB cap).
    local A_SIZE=$(stat -f%z "$DIST/libsofuu-linux-$ARCH.a" 2>/dev/null || stat -c%s "$DIST/libsofuu-linux-$ARCH.a" 2>/dev/null || echo 0)
    if [ "$A_SIZE" -gt 5242880 ]; then
        echo "  \033[31m✗ Static lib exceeds 5MB cap ($A_SIZE bytes)\033[0m"
        exit 1
    fi
}

# Copy header.
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h" 2>/dev/null || true
mkdir -p "$DIST"

case "$ARCH_ARG" in
    x86_64|arm64) build_arch "$ARCH_ARG" ;;
    all) build_arch "x86_64"; build_arch "arm64" ;;
    *) echo "Usage: $0 [x86_64|arm64|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mDone. Artifacts in dist/:\033[0m"
ls -lh "$DIST"/libsofuu-linux-* "$DIST/sofuu_embed.h" 2>/dev/null || true
