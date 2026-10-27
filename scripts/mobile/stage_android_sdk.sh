#!/usr/bin/env bash
# scripts/mobile/stage_android_sdk.sh — stage the QTSQ-free Android pack
# into the com.sofuu:sofuu-android AAR.
#
# `make dist-android` produces dist/libsofuu-android-{aarch64,x86_64}.so.
# This script copies them to the AAR's jniLibs, adds the NDK's
# libc++_shared.so (libsofuu declares it as NEEDED), and keeps the C header
# copy in sync.
#
# The generated .so files are gitignored; this is a release step, not source.
#
# Usage: bash scripts/mobile/stage_android_sdk.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
JNI="$REPO_ROOT/bindings/android/sofuu/src/main/jniLibs"
HEADER_OUT="$REPO_ROOT/bindings/android/sofuu/src/main/cpp/include"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/mobile/ndk_env.sh"

stage_abi() {
    local src_abi="$1"
    local ndk_abi="$2"
    local rust_target="$3"
    local src="$DIST/libsofuu-android-$src_abi.so"

    [ -f "$src" ] || {
        echo "✗ missing $src — run 'make dist-android' first"
        exit 1
    }
    mkdir -p "$JNI/$ndk_abi"
    cp "$src" "$JNI/$ndk_abi/libsofuu.so"
    echo "✓ $JNI/$ndk_abi/libsofuu.so ($(du -h "$src" | cut -f1))"

    # Every Rust cdylib on Android needs the shared C++ runtime. The AAR's
    # CMake target is pure C, so AGP will not pull it in by itself.
    local cxx="$MOBILE_NDK_HOME/toolchains/llvm/prebuilt/$MOBILE_HOST_TAG/sysroot/usr/lib/$rust_target/libc++_shared.so"
    [ -f "$cxx" ] || {
        echo "✗ libc++_shared.so not found at $cxx"
        exit 1
    }
    cp "$cxx" "$JNI/$ndk_abi/libc++_shared.so"
    echo "✓ $JNI/$ndk_abi/libc++_shared.so"
}

stage_abi aarch64 arm64-v8a aarch64-linux-android
stage_abi x86_64 x86_64 x86_64-linux-android

mkdir -p "$HEADER_OUT"
cp "$REPO_ROOT/include/sofuu_embed.h" "$HEADER_OUT/sofuu_embed.h"
echo "✓ $HEADER_OUT/sofuu_embed.h"

echo ""
echo "Staged AAR payload. Next:"
echo "  cd bindings/android && ./gradlew :sofuu:assembleRelease"
