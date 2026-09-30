#!/usr/bin/env bash
# scripts/dist/ios.sh — H4 platform pack: iOS (arm64 device + simulator)
#
# Produces: dist/libsofuu-ios-arm64.a (device)
#           dist/libsofuu-ios-sim.a   (simulator, x86_64 + arm64 universal)
#           dist/libsofuu.xcframework (combined device + sim)
#           dist/sofuu_embed.h
#
# Usage: bash scripts/dist/ios.sh [target]
#   target: device | sim | all | xcframework (default: all)
#
# Prerequisites: Xcode, Rust toolchain with iOS targets.
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
#
# QuickJS is a pure interpreter (no JIT) → iOS App Store executable-memory
# policy is satisfied by design. This is the key iOS architectural advantage.
#
# NOTE: QTSQ-free build — the proprietary codec checkout is macOS-only.
# iOS packs ship pure MIT.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
TARGET_ARG="${1:-all}"

build_target() {
    local NAME="$1"
    local RUST_TARGET="$2"
    local OUT_NAME="$3"

    echo ""
    echo "  \033[36mBuilding libsofuu for iOS $NAME ($RUST_TARGET)\033[0m"

    cd "$REPO_ROOT"

    if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
        echo "  Installing rustup target $RUST_TARGET..."
        rustup target add "$RUST_TARGET"
    fi

    # Build libuv for the target.
    local UV_DIR="deps/libuv"
    local UV_BUILD="deps/libuv/build-ios-$NAME"
    local IOS_SYSROOT=""
    case "$NAME" in
        device) IOS_SYSROOT="$(xcrun --sdk iphoneos --show-sdk-path 2>/dev/null || echo '')" ;;
        sim)    IOS_SYSROOT="$(xcrun --sdk iphonesimulator --show-sdk-path 2>/dev/null || echo '')" ;;
    esac

    local CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTING=OFF"
    if [ -n "$IOS_SYSROOT" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_OSX_SYSROOT=$IOS_SYSROOT"
    fi

    cmake -S "$UV_DIR" -B "$UV_BUILD" -G Ninja $CMAKE_ARGS 2>&1 | tail -3
    cmake --build "$UV_BUILD" --target uv_a 2>&1 | tail -3

    export SOFUU_UV_DIR="$UV_BUILD"
    mkdir -p "$DIST/no-qtsq"
    export SOFUU_QTSQ_DIR="$DIST/no-qtsq"

    cargo build --release -p sofuu-capi --target "$RUST_TARGET"

    mkdir -p "$DIST"
    cp "target/${RUST_TARGET}/release/libsofuu_capi.a" "$DIST/$OUT_NAME" 2>/dev/null || \
        cp "target/${RUST_TARGET}/release/libsofuu.a" "$DIST/$OUT_NAME" 2>/dev/null || true

    local SIZE=$(du -sh "$DIST/$OUT_NAME" 2>/dev/null | cut -f1)
    echo "  \033[32m✓\033[0m $OUT_NAME ($SIZE)"

    # Size check (5MB cap).
    local A_SIZE=$(stat -f%z "$DIST/$OUT_NAME" 2>/dev/null || echo 0)
    if [ "$A_SIZE" -gt 5242880 ]; then
        echo "  \033[31m✗ Static lib exceeds 5MB cap ($A_SIZE bytes)\033[0m"
        exit 1
    fi
}

build_xcframework() {
    echo ""
    echo "  \033[36mCreating xcframework...\033[0m"

    # Create a minimal header wrapper for the xcframework.
    local HEADER_DIR="$DIST/libsofuu-headers"
    mkdir -p "$HEADER_DIR"
    cp "$REPO_ROOT/include/sofuu_embed.h" "$HEADER_DIR/"

    # Create module map for Swift consumption.
    cat > "$HEADER_DIR/module.modulemap" << 'EOF'
module Sofuu {
    umbrella header "sofuu_embed.h"
    export *
    module * { export * }
}
EOF

    local DEVICE_LIB="$DIST/libsofuu-ios-arm64.a"
    local SIM_LIB="$DIST/libsofuu-ios-sim.a"

    if [ ! -f "$DEVICE_LIB" ] || [ ! -f "$SIM_LIB" ]; then
        echo "  \033[33m⚠ Missing device or sim lib — building both first...\033[0m"
        build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a"
        build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a"
    fi

    # Create the xcframework.
    rm -rf "$DIST/libsofuu.xcframework"
    xcodebuild -create-xcframework \
        -library "$DEVICE_LIB" -headers "$HEADER_DIR" \
        -library "$SIM_LIB" -headers "$HEADER_DIR" \
        -output "$DIST/libsofuu.xcframework" 2>&1 | tail -5

    if [ -d "$DIST/libsofuu.xcframework" ]; then
        echo "  \033[32m✓\033[0m libsofuu.xcframework"
        du -sh "$DIST/libsofuu.xcframework"
    else
        echo "  \033[31m✗ xcframework creation failed\033[0m"
        exit 1
    fi
}

# Copy header.
mkdir -p "$DIST"
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h" 2>/dev/null || true

case "$TARGET_ARG" in
    device) build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a" ;;
    sim)    build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a" ;;
    xcframework) build_xcframework ;;
    all)
        build_target "device" "aarch64-apple-ios" "libsofuu-ios-arm64.a"
        build_target "sim" "aarch64-apple-ios-sim" "libsofuu-ios-sim.a"
        build_xcframework
        ;;
    *) echo "Usage: $0 [device|sim|xcframework|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mDone. iOS artifacts in dist/:\033[0m"
ls -lh "$DIST"/libsofuu-ios-* "$DIST/sofuu_embed.h" 2>/dev/null || true
if [ -d "$DIST/libsofuu.xcframework" ]; then
    echo "  xcframework: $(du -sh "$DIST/libsofuu.xcframework" | cut -f1)"
fi
