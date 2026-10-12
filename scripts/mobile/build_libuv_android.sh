#!/usr/bin/env bash
# scripts/mobile/build_libuv_android.sh — static libuv per Android ABI.
# Produces: dist/libuv-android-<abi>/libuv.a
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ndk_env.sh" "$@"

build_one() {
    local abi="$1"
    mobile_abi_env "$abi"
    local out="$REPO_ROOT/dist/libuv-android-$MOBILE_ABI_NAME"
    local build="$REPO_ROOT/deps/libuv/build-android-$MOBILE_ABI_NAME"
    if [ -f "$out/libuv.a" ]; then
        echo "✓ libuv $MOBILE_ABI_NAME already built"
        return 0
    fi
    echo "── libuv → $MOBILE_ABI_NAME"
    cmake -S "$REPO_ROOT/deps/libuv" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_TOOLCHAIN_FILE="$MOBILE_CMAKE_TOOLCHAIN" \
        -DANDROID_ABI="$MOBILE_ABI_NAME" \
        -DANDROID_PLATFORM="android-$MOBILE_API" \
        -DANDROID_NDK="$MOBILE_NDK_HOME" >/dev/null 2>&1
    cmake --build "$build" --target uv_a >/dev/null 2>&1
    mkdir -p "$out"
    find "$build" -name "libuv.a" -exec cp {} "$out/libuv.a" \;
    echo "✓ libuv $MOBILE_ABI_NAME → $out/libuv.a"
}

# "all" (the Makefile's spelling) and an empty list both mean every ABI.
ABIS=()
for a in "$@"; do [ "$a" = "all" ] || ABIS+=("$a"); done
if [ ${#ABIS[@]} -eq 0 ]; then
    build_one arm64
    build_one x86_64
else
    for abi in "${ABIS[@]}"; do build_one "$abi"; done
fi
