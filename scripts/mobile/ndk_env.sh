#!/usr/bin/env bash
# scripts/mobile/ndk_env.sh — shared NDK plumbing for the Android engine
# cross-builds. Source it:  source scripts/mobile/ndk_env.sh <abi>
#
# Resolves the NDK, exports the clang toolchain, and defines the per-ABI
# identity (rust triple, NDK clang prefix, android platform) used by the
# library + engine build scripts.

MOBILE_NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/27.0.12077973}}"
if [ ! -d "$MOBILE_NDK_HOME" ]; then
    # Fall back to the newest installed NDK. Callers run with `set -o
    # pipefail`; a failing `ls` inside the substitution must not kill this
    # (sourced) script before the "✗ NDK not found" message below — `ls`
    # exits 2 when the glob matches nothing.
    MOBILE_NDK_HOME="$(ls -d "$HOME"/Library/Android/sdk/ndk/*/ 2>/dev/null | sort -V | tail -1 || true)"
    MOBILE_NDK_HOME="${MOBILE_NDK_HOME%/}"
fi
[ -d "$MOBILE_NDK_HOME" ] || { echo "✗ NDK not found (set ANDROID_NDK_HOME)"; exit 1; }

MOBILE_HOST_TAG="darwin-x86_64"
case "$(uname -s)" in
    Darwin) MOBILE_HOST_TAG="darwin-x86_64" ;;
    Linux)  MOBILE_HOST_TAG="linux-x86_64" ;;
    *) echo "✗ unsupported host for NDK: $(uname -s)"; exit 1 ;;
esac
MOBILE_NDK_BIN="$MOBILE_NDK_HOME/toolchains/llvm/prebuilt/$MOBILE_HOST_TAG/bin"
[ -x "$MOBILE_NDK_BIN/llvm-ar" ] || { echo "✗ NDK toolchain missing at $MOBILE_NDK_BIN"; exit 1; }

MOBILE_API="${ANDROID_API_LEVEL:-24}"

# Ninja: not on PATH by default on macOS. The Android SDK ships it next to
# its cmake (and that's also the cmake AGP uses for the app's own build).
SDK_CMAKE_BIN="$(ls -d "$HOME"/Library/Android/sdk/cmake/*/bin 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$SDK_CMAKE_BIN" ] && [ -x "$SDK_CMAKE_BIN/ninja" ]; then
    export PATH="$SDK_CMAKE_BIN:$PATH"
fi

mobile_abi_env() {
    local abi="$1"
    case "$abi" in
        arm64|aarch64|arm64-v8a)
            MOBILE_ABI_NAME="arm64-v8a"
            MOBILE_RUST_TARGET="aarch64-linux-android"
            MOBILE_NDK_PREFIX="aarch64-linux-android$MOBILE_API"
            MOBILE_OPENSSL_TARGET="android-arm64"
            ;;
        x86_64|x86_64-64)
            MOBILE_ABI_NAME="x86_64"
            MOBILE_RUST_TARGET="x86_64-linux-android"
            MOBILE_NDK_PREFIX="x86_64-linux-android$MOBILE_API"
            MOBILE_OPENSSL_TARGET="android-x86_64"
            ;;
        *) echo "✗ unknown ABI: $abi (arm64|x86_64)"; exit 1 ;;
    esac
    MOBILE_CC="$MOBILE_NDK_BIN/${MOBILE_NDK_PREFIX}-clang"
    MOBILE_CXX="$MOBILE_NDK_BIN/${MOBILE_NDK_PREFIX}-clang++"
    MOBILE_AR="$MOBILE_NDK_BIN/llvm-ar"
    MOBILE_RANLIB="$MOBILE_NDK_BIN/llvm-ranlib"
    [ -x "$MOBILE_CC" ] || { echo "✗ $MOBILE_CC not executable"; exit 1; }

    # Put the NDK toolchain on PATH: OpenSSL's Configure resolves `clang`
    # through PATH and requires it to live inside the NDK; cargo's default
    # linker name ("cc") then also resolves to the NDK clang, which makes
    # the CARGO_TARGET_*_LINKER export below belt-and-braces.
    export PATH="$MOBILE_NDK_BIN:$PATH"

    # OpenSSL's Configure reads ANDROID_NDK_ROOT (not ANDROID_NDK_HOME).
    export ANDROID_NDK_ROOT="$MOBILE_NDK_HOME"
    export ANDROID_NDK_HOME="$MOBILE_NDK_HOME"

    # The cc crate (libsofuu's C build) reads CC/CXX; the final Rust link
    # needs the linker named explicitly — cargo's default "cc" is the host
    # compiler. Both are exported for every cargo invocation.
    export CC="$MOBILE_CC" CXX="$MOBILE_CXX" AR="$MOBILE_AR" RANLIB="$MOBILE_RANLIB"
    local linker_var="CARGO_TARGET_$(echo "$MOBILE_RUST_TARGET" | tr 'a-z-' 'A-Z_')_LINKER"
    export "$linker_var=$MOBILE_CC"

    # The NDK cmake toolchain file drives libuv/curl cross builds.
    MOBILE_CMAKE_TOOLCHAIN="$MOBILE_NDK_HOME/build/cmake/android.toolchain.cmake"
    [ -f "$MOBILE_CMAKE_TOOLCHAIN" ] || { echo "✗ cmake toolchain missing: $MOBILE_CMAKE_TOOLCHAIN"; exit 1; }
}

# Sourced WITH an ABI argument → resolve it immediately (scripts source
# with "$@"); sourced with no args → only the helper is defined.
if [ "$#" -gt 0 ]; then
    mobile_abi_env "$1"
fi
