#!/usr/bin/env bash
# scripts/mobile/build_engine_android.sh — build the QTSQ-linked
# libsofuu .so for each Android ABI and stage it into the app's jniLibs.
#
# Pipeline per ABI (all cached under dist/):
#   1. libuv   → dist/libuv-android-<abi>/libuv.a
#   2. OpenSSL → dist/openssl-android-<abi>/{libssl.a,libcrypto.a,include}
#   3. curl    → dist/libcurl-android-<abi>/lib/libcurl.a   (OpenSSL TLS)
#   4. QTSQ    → dist/qtsq-android-<abi>/{libqtsq.a,compressor/libqtc.a,libz.a,include}
#   5. cargo   → target/<triple>/release/libsofuu_capi.so
#   6. stage   → sofuu-android/app/src/main/jniLibs/<abi>/libsofuu.so
#
# The public SDK (make dist-android) stays QTSQ-free/MIT-clean; THIS build
# links QTSQ so the app gets the same encrypted brain + session store the
# desktop has (the owner's QTSQ checkout + grant covers it).
#
# Usage: bash scripts/mobile/build_engine_android.sh [arm64|x86_64 …]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Expand "all" to the real ABI list ONCE, up front. The dependency scripts
# below source ndk_env.sh with each argument and that rejects anything it
# does not recognise — so passing the literal "all" straight through (which
# is what `make mobile-engine` does) aborts with "unknown ABI: all" before
# a single library is configured.
ABIS=()
for abi in "$@"; do
    if [ "$abi" = "all" ]; then
        ABIS+=(arm64 x86_64)
    else
        ABIS+=("$abi")
    fi
done
[ ${#ABIS[@]} -eq 0 ] && ABIS=(arm64 x86_64)
set -- "${ABIS[@]}"

bash "$REPO_ROOT/scripts/mobile/build_libuv_android.sh" "$@"
bash "$REPO_ROOT/scripts/mobile/build_openssl_android.sh" "$@"
bash "$REPO_ROOT/scripts/mobile/build_libcurl_android.sh" "$@"
bash "$REPO_ROOT/scripts/mobile/build_qtsq_android.sh" "$@"

# Profile: SOFUU_ANDROID_PROFILE=debug for the fast smoke loop (default),
# release for shipping. RUSTFLAGS changes also invalidate the cached build.
CARGO_PROFILE="${SOFUU_ANDROID_PROFILE:-debug}"
case "$CARGO_PROFILE" in
    debug) PROFILE_FLAG=""; CARGO_PROFILE_DIR="debug" ;;
    release) PROFILE_FLAG="--release"; CARGO_PROFILE_DIR="release" ;;
    *) echo "✗ unknown profile: $CARGO_PROFILE"; exit 1 ;;
esac

for abi in "${ABIS[@]}"; do
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/mobile/ndk_env.sh" "$abi"

    echo "── libsofuu (sofuu-capi) → $MOBILE_ABI_NAME ($MOBILE_RUST_TARGET)"
    rustup target list --installed 2>/dev/null | grep -q "^${MOBILE_RUST_TARGET}\$" || {
        echo "→ adding rustup target $MOBILE_RUST_TARGET"
        rustup target add "$MOBILE_RUST_TARGET"
    }

    # The SONAME matters: a Rust cdylib carries no DT_SONAME by default, so
    # the app's CMake link would bake the build-time ABSOLUTE path into
    # libsofuu_jni.so's DT_NEEDED and the APK would fail to load on device.
    # With -Wl,-soname the app links against a plain "libsofuu.so" that
    # resolves from the APK's lib dir at runtime.
    #
    # max-page-size=16384 is equally load-bearing: the NDK toolchain
    # defaults every PT_LOAD to 4 KiB, and targetSdk 35 means the APK must
    # load on Android 15+ devices with a 16 KiB page size (most current
    # Pixels). Without this the linker only aligns the FIRST segment and
    # the .so is rejected at load time. The app's own libsofuu_jni.so needs
    # the same flag on its target (see app/src/main/cpp/CMakeLists.txt).
    (
        cd "$REPO_ROOT"
        SOFUU_UV_DIR="$REPO_ROOT/dist/libuv-android-$MOBILE_ABI_NAME" \
        SOFUU_CURL_STATIC_DIR="$REPO_ROOT/dist/libcurl-android-$MOBILE_ABI_NAME/lib" \
        SOFUU_QTSQ_DIR="$REPO_ROOT/dist/qtsq-android-$MOBILE_ABI_NAME" \
        SOFUU_ZLIB_DIR="$REPO_ROOT/dist/qtsq-android-$MOBILE_ABI_NAME" \
        RUSTFLAGS="-C link-arg=-Wl,-soname,libsofuu.so -C link-arg=-Wl,-z,max-page-size=16384" \
        cargo build $PROFILE_FLAG -p sofuu-capi --target "$MOBILE_RUST_TARGET"
    )

    local_so="target/${MOBILE_RUST_TARGET}/$CARGO_PROFILE_DIR/libsofuu_capi.so"
    [ -f "$REPO_ROOT/$local_so" ] || local_so="target/${MOBILE_RUST_TARGET}/$CARGO_PROFILE_DIR/libsofuu.so"
    [ -f "$REPO_ROOT/$local_so" ] || { echo "✗ build produced no libsofuu .so"; exit 1; }

    stage="$REPO_ROOT/sofuu-android/app/src/main/jniLibs/$MOBILE_ABI_NAME"
    mkdir -p "$stage"
    cp "$REPO_ROOT/$local_so" "$stage/libsofuu.so"
    size=$(du -h "$stage/libsofuu.so" | cut -f1)
    echo "✓ staged $stage/libsofuu.so ($size)"
done

echo ""
echo "Engine libs built and staged. Next: cd sofuu-android && ./gradlew :app:assembleDebug"
