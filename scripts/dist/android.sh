#!/usr/bin/env bash
# scripts/dist/android.sh — H4 platform pack: Android (aarch64/armv7/x86_64)
#
# Produces: dist/libsofuu-android-{aarch64,armv7,x86_64}.so
#           dist/sofuu_embed.h
#
# Usage: bash scripts/dist/android.sh [abi]
#   abi: aarch64 | armv7 | x86_64 | all (default: all)
#
# Prerequisites: Android NDK (ANDROID_NDK_HOME or ANDROID_NDK set) and the
# per-ABI QTSQ archives under dist/qtsq-android-<abi>/.
#   rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
#
# QuickJS is a pure interpreter → no JIT → Android compatible.
# QTSQ is a core component; a codec-less build is rejected.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO_ROOT/dist"
ABI_ARG="${1:-all}"

# Locate the NDK.
NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
if [ -z "$NDK_HOME" ]; then
    echo "  \033[31m✗ ANDROID_NDK_HOME not set\033[0m"
    echo "  Set it to your NDK root, e.g.:"
    echo "    export ANDROID_NDK_HOME=/path/to/Android/Sdk/ndk/27.0.12077973"
    exit 1
fi
if [ ! -d "$NDK_HOME" ]; then
    echo "  \033[31m✗ NDK directory not found: $NDK_HOME\033[0m"
    exit 1
fi

NDK_TOOLCHAIN="$NDK_HOME/toolchains/llvm/prebuilt"
HOST_TAG=""
case "$(uname -s)" in
    Darwin) HOST_TAG="darwin-x86_64" ;;
    Linux)  HOST_TAG="linux-x86_64" ;;
    *) echo "Unknown host OS for NDK"; exit 1 ;;
esac
NDK_CC="$NDK_TOOLCHAIN/$HOST_TAG/bin"

build_abi() {
    local ABI="$1"
    local RUST_TARGET=""
    local API_LEVEL="${ANDROID_API_LEVEL:-24}"  # minSdk 24
    local NDK_PREFIX=""
    local CMAKE_ABI=""

    case "$ABI" in
        aarch64)
            RUST_TARGET="aarch64-linux-android"
            NDK_PREFIX="aarch64-linux-android${API_LEVEL}"
            CMAKE_ABI="arm64-v8a"
            ;;
        armv7)
            RUST_TARGET="armv7-linux-androideabi"
            NDK_PREFIX="armv7a-linux-androideabi${API_LEVEL}"
            CMAKE_ABI="armeabi-v7a"
            ;;
        x86_64)
            RUST_TARGET="x86_64-linux-android"
            NDK_PREFIX="x86_64-linux-android${API_LEVEL}"
            CMAKE_ABI="x86_64"
            ;;
        *) echo "Unknown ABI: $ABI"; exit 1 ;;
    esac

    echo ""
    echo "  \033[36mBuilding libsofuu for Android $ABI ($RUST_TARGET)\033[0m"

    cd "$REPO_ROOT"

    if ! rustup target list --installed 2>/dev/null | grep -q "^${RUST_TARGET}$"; then
        echo "  Installing rustup target $RUST_TARGET..."
        rustup target add "$RUST_TARGET"
    fi

    # Build libuv for the target.
    local UV_DIR="deps/libuv"
    local UV_BUILD="deps/libuv/build-android-$ABI"
    export CC="$NDK_CC/${NDK_PREFIX}-clang"
    export CXX="$NDK_CC/${NDK_PREFIX}-clang++"
    export AR="$NDK_CC/llvm-ar"
    export RANLIB="$NDK_CC/llvm-ranlib"
    # Cargo's default linker is the host `cc`; name the NDK clang for the
    # Rust target explicitly. Without this the cross build reaches the final
    # link, then a macOS ld rejects the Android linker flags.
    local linker_var="CARGO_TARGET_$(echo "$RUST_TARGET" | tr 'a-z-' 'A-Z_')_LINKER"
    export "$linker_var=$NDK_CC/${NDK_PREFIX}-clang"

    cmake -S "$UV_DIR" -B "$UV_BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_SYSTEM_NAME=Android \
        -DCMAKE_ANDROID_ARCH_ABI="$CMAKE_ABI" \
        -DCMAKE_ANDROID_API="$API_LEVEL" \
        -DCMAKE_ANDROID_NDK="$NDK_HOME" \
        2>&1 | tail -3
    cmake --build "$UV_BUILD" --target uv_a 2>&1 | tail -3

    # The Android runtime needs a TLS-capable static libcurl. The mobile
    # scripts build it with OpenSSL; the old dist path silently skipped this
    # and then failed at the final link with "unable to find library -lcurl".
    local CURL_DIR="$DIST/libcurl-android-$CMAKE_ABI/lib"
    if [ ! -f "$CURL_DIR/libcurl.a" ]; then
        echo "  ✗ libcurl missing for $CMAKE_ABI at $CURL_DIR"
        echo "    run scripts/mobile/build_openssl_android.sh and"
        echo "    scripts/mobile/build_libcurl_android.sh first"
        exit 1
    fi
    export SOFUU_CURL_STATIC_DIR="$CURL_DIR"

    export SOFUU_UV_DIR="$REPO_ROOT/$UV_BUILD"

    # QTSQ is core. The per-ABI archive comes from
    # scripts/mobile/build_qtsq_android.sh; without it the build would
    # silently produce a memory-less runtime, so fail closed.
    local QTSQ_DIR="$DIST/qtsq-android-$CMAKE_ABI"
    if [ ! -f "$QTSQ_DIR/libqtsq.a" ] || [ ! -f "$QTSQ_DIR/compressor/libqtc.a" ]; then
        echo "  ✗ QTSQ archives missing for $CMAKE_ABI at $QTSQ_DIR"
        echo "    run scripts/mobile/build_qtsq_android.sh $ABI first"
        exit 1
    fi
    export SOFUU_QTSQ_DIR="$QTSQ_DIR"
    export SOFUU_ZLIB_DIR="$QTSQ_DIR"

    cargo build --release -p sofuu-capi --target "$RUST_TARGET"

    mkdir -p "$DIST"
    cp "target/${RUST_TARGET}/release/libsofuu_capi.so" "$DIST/libsofuu-android-$ABI.so" 2>/dev/null || \
        cp "target/${RUST_TARGET}/release/libsofuu.so" "$DIST/libsofuu-android-$ABI.so" 2>/dev/null || true

    local SIZE=$(du -sh "$DIST/libsofuu-android-$ABI.so" 2>/dev/null | cut -f1)
    echo "  \033[32m✓\033[0m libsofuu-android-$ABI.so ($SIZE)"

    # Size budget. Android includes the in-process inference path (GGUF /
    # SafeTensors) plus HTTP/TLS, so it does not fit the 5 MB desktop-pack
    # cap. Keep a hard ceiling and allow an operator override for a slimmer
    # feature build.
    local MAX_SIZE="${SOFUU_ANDROID_MAX_SIZE:-12582912}"
    local SO_SIZE=$(stat -f%z "$DIST/libsofuu-android-$ABI.so" 2>/dev/null || stat -c%s "$DIST/libsofuu-android-$ABI.so" 2>/dev/null || echo 0)
    if [ "$SO_SIZE" -gt "$MAX_SIZE" ]; then
        echo "  \033[31m✗ Shared lib exceeds Android size cap ($SO_SIZE > $MAX_SIZE bytes)\033[0m"
        exit 1
    fi
}

# Copy header.
mkdir -p "$DIST"
cp "$REPO_ROOT/include/sofuu_embed.h" "$DIST/sofuu_embed.h" 2>/dev/null || true

case "$ABI_ARG" in
    aarch64|armv7|x86_64) build_abi "$ABI_ARG" ;;
    all) build_abi "aarch64"; build_abi "x86_64" ;;
    *) echo "Usage: $0 [aarch64|armv7|x86_64|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mDone. Android artifacts in dist/:\033[0m"
ls -lh "$DIST"/libsofuu-android-* "$DIST/sofuu_embed.h" 2>/dev/null || true
