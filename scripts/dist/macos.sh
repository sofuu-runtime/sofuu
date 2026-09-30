#!/usr/bin/env bash
# scripts/dist/macos.sh — H4 platform pack: macOS (arm64 + x86_64)
#
# Produces: dist/libsofuu-darwin-{arm64,x86_64}.a
#           dist/libsofuu-darwin-{arm64,x86_64}.dylib
#           dist/sofuu_embed.h
#           dist/libsofuu-darwin-{arm64,x86_64}.tar.gz
#
# Usage: bash scripts/dist/macos.sh [arch]
#   arch: arm64 | x86_64 | all (default: current arch)
#
# Prerequisites: Rust toolchain, cmake, ninja, libcurl (brew install curl cmake ninja)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
ARCH_ARG="${1:-$(uname -m)}"

# macOS reports arm64/x86_64 — normalize.
if [ "$ARCH_ARG" = "aarch64" ]; then ARCH_ARG="arm64"; fi

build_arch() {
    local ARCH="$1"
    local RUST_TARGET=""
    case "$ARCH" in
        arm64)   RUST_TARGET="aarch64-apple-darwin" ;;
        x86_64)  RUST_TARGET="x86_64-apple-darwin" ;;
        *) echo "Unknown arch: $ARCH"; exit 1 ;;
    esac

    echo ""
    echo "  \033[36mBuilding libsofuu for macOS $ARCH ($RUST_TARGET)\033[0m"

    cd "$REPO_ROOT"

    # Ensure the rustup target is installed.
    if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
        rustup target add "$RUST_TARGET"
    fi

    # Build libuv for the target (PIC for shared lib).
    local UV_DIR="deps/libuv"
    local UV_BUILD="deps/libuv/build-macos-$ARCH"
    cmake -S "$UV_DIR" -B "$UV_BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
        2>&1 | tail -3
    cmake --build "$UV_BUILD" --target uv_a 2>&1 | tail -3

    export SOFUU_UV_DIR="$UV_BUILD"
    # QTSQ-free (CI convention).
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"

    cargo build --release -p sofuu-capi --target "$RUST_TARGET"

    # Copy artifacts.
    mkdir -p "$DIST"
    local LIB_DIR="target/${RUST_TARGET}/release"
    cp "$LIB_DIR/libsofuu_capi.a" "$DIST/libsofuu-darwin-$ARCH.a" 2>/dev/null || \
        cp "$LIB_DIR/libsofuu.a" "$DIST/libsofuu-darwin-$ARCH.a" 2>/dev/null || true
    cp "$LIB_DIR/libsofuu_capi.dylib" "$DIST/libsofuu-darwin-$ARCH.dylib" 2>/dev/null || \
        cp "$LIB_DIR/libsofuu.dylib" "$DIST/libsofuu-darwin-$ARCH.dylib" 2>/dev/null || true

    # Create tarball with header.
    local TARBALL_DIR="$DIST/libsofuu-darwin-$ARCH"
    mkdir -p "$TARBALL_DIR"
    cp "$DIST/libsofuu-darwin-$ARCH.a" "$TARBALL_DIR/" 2>/dev/null || true
    cp "$DIST/libsofuu-darwin-$ARCH.dylib" "$TARBALL_DIR/" 2>/dev/null || true
    cp include/sofuu_embed.h "$TARBALL_DIR/"
    tar -czf "$DIST/libsofuu-darwin-$ARCH.tar.gz" -C "$DIST" "$(basename $TARBALL_DIR)"
    rm -rf "$TARBALL_DIR"

    local SIZE=$(du -sh "$DIST/libsofuu-darwin-$ARCH.a" 2>/dev/null | cut -f1)
    echo "  \033[32m✓\033[0m libsofuu-darwin-$ARCH.a ($SIZE)"
    local DSIZE=$(du -sh "$DIST/libsofuu-darwin-$ARCH.dylib" 2>/dev/null | cut -f1)
    echo "  \033[32m✓\033[0m libsofuu-darwin-$ARCH.dylib ($DSIZE)"

    # Size check (5MB cap).
    local A_SIZE=$(stat -f%z "$DIST/libsofuu-darwin-$ARCH.a" 2>/dev/null || stat -c%s "$DIST/libsofuu-darwin-$ARCH.a" 2>/dev/null || echo 0)
    if [ "$A_SIZE" -gt 5242880 ]; then
        echo "  \033[31m✗ Static lib exceeds 5MB cap ($A_SIZE bytes)\033[0m"
        exit 1
    fi
}

# Copy header.
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h" 2>/dev/null || true
mkdir -p "$DIST"

case "$ARCH_ARG" in
    arm64|x86_64) build_arch "$ARCH_ARG" ;;
    all) build_arch "arm64"; build_arch "x86_64" ;;
    *) echo "Usage: $0 [arm64|x86_64|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mDone. Artifacts in dist/:\033[0m"
ls -lh "$DIST"/libsofuu-darwin-* "$DIST/sofuu_embed.h" 2>/dev/null || true
