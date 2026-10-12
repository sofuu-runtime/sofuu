#!/usr/bin/env bash
# scripts/mobile/build_openssl_ios.sh — static OpenSSL per iOS slice.
#
# iOS has no system OpenSSL and the SDK ships no libcurl stub, so the
# runtime's TLS stack has to come with the app. This is the iOS twin of
# build_openssl_android.sh: the same OpenSSL version, the same pinned hash,
# the same no-shared/no-tests shape — only the Configure target changes
# (ios64-xcrun / iossimulator-arm64-xcrun, which drive `xcrun -sdk … cc`).
#
# Note on SecureTransport: curl 8.6 has no SecureTransport backend any more
# (removed in curl 8.0), so "just use Apple's TLS" is not an option. OpenSSL
# is the path, and it matches what Android already ships.
#
# Produces: dist/openssl-ios-<slice>/lib/{libssl.a,libcrypto.a} (+ include)
#
# Usage: bash scripts/mobile/build_openssl_ios.sh [device|sim|all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ios_env.sh"

OPENSSL_VER="3.3.2"
# Same pin as build_openssl_android.sh — one version, one hash, both platforms.
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VER}/openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SHA256="2e8a40b01979afe8be0bbfb3de5dc1c6709fedb46d6c89c10da114ab5fc3d281"
OPENSSL_TARBALL="openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SRC="/tmp/openssl-${OPENSSL_VER}"

build_one() {
    local slice="$1"
    ios_slice_env "$slice"

    local out="$REPO_ROOT/dist/openssl-ios-$IOS_SLICE_NAME"
    if [ -f "$out/lib/libssl.a" ] && [ -f "$out/lib/libcrypto.a" ]; then
        echo "✓ openssl $IOS_SLICE_NAME already built"
        return 0
    fi

    if [ ! -f "$OPENSSL_SRC/Configure" ]; then
        echo "→ downloading openssl ${OPENSSL_VER} (official release asset)"
        rm -f "/tmp/$OPENSSL_TARBALL"
        curl -fsSL --max-time 180 -o "/tmp/$OPENSSL_TARBALL" "$OPENSSL_URL"
        local got
        got="$(shasum -a 256 "/tmp/$OPENSSL_TARBALL" | cut -d' ' -f1)"
        [ "$got" = "$OPENSSL_SHA256" ] || {
            echo "✗ openssl tarball checksum mismatch"
            echo "  expected $OPENSSL_SHA256"
            echo "  got      $got"
            exit 1
        }
        rm -rf "$OPENSSL_SRC"
        tar xzf "/tmp/$OPENSSL_TARBALL" -C /tmp
    fi

    echo "── openssl ${OPENSSL_VER} → $IOS_SLICE_NAME ($IOS_OPENSSL_TARGET, $IOS_ARCH)"
    local build="$OPENSSL_SRC/build-ios-$IOS_SLICE_NAME"
    rm -rf "$build"
    mkdir -p "$build" "$out/lib" "$out/include"
    (
        cd "$build"
        # ── Unset CC/CXX/AR/RANLIB, deliberately ─────────────────────
        # OpenSSL's Configure DOES honour an inherited $CC: with it set the
        # generated Makefile reads
        #     CC=$(CROSS_COMPILE)/…/XcodeDefault.xctoolchain/usr/bin/clang
        # — the raw Xcode clang, with NO `-isysroot`. That clang defaults to
        # `Target: arm64-apple-darwin…`, i.e. it silently builds OpenSSL for
        # **macOS**. The `-xcrun` targets exist precisely to avoid this: their
        # CC is `xcrun -sdk iphoneos cc`, and xcrun is what injects the
        # sysroot. So let the target own the compiler.
        # (Verified: clean env → `CC=$(CROSS_COMPILE)xcrun -sdk
        # iphonesimulator cc`, and CNF_CFLAGS keeps `-arch arm64 -fno-common`.)
        #
        # Note $AR/$RANLIB are NOT honoured by Configure (darwin-common pins
        # them to `ar`/`ranlib`), so there is nothing to restore there — the
        # host binutils handle arm64 Mach-O fine.
        #
        # `no-asm` is deliberately NOT passed: the arm64 perlasm works fine
        # under xcrun and the asm gives a real speed-up on the AES paths that
        # dominate a TLS handshake.
        #
        # $IOS_MIN_FLAG is per-slice and NOT optional. Without it clang uses
        # the SDK's own default (minos 26.2 on Xcode 26.2), which then makes
        # every app link emit "object file was built for newer iOS version
        # than being linked". And it must be the *simulator* spelling for the
        # sim slice — `-mios-version-min` there silently yields platform IOS.
        env -u CC -u CXX -u AR -u RANLIB \
            PERL="${PERL:-perl}" "$OPENSSL_SRC/Configure" \
            "$IOS_OPENSSL_TARGET" \
            no-shared no-tests no-module \
            "$IOS_MIN_FLAG" \
            --prefix="$out" --openssldir="$out" \
            >"$build/openssl.log" 2>&1
        make -s -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" build_libs \
            >>"$build/openssl.log" 2>&1
        cp libssl.a libcrypto.a "$out/lib/"
        # Public headers = the checkout's include tree with the build tree's
        # GENERATED headers (ssl.h from ssl.h.in, opensslconf.h, …) overlaid.
        cp -R "$OPENSSL_SRC/include/." "$out/include/"
        cp -R "$build/include/." "$out/include/" 2>/dev/null || true
    ) || {
        # Configure/make are quiet by design; surface the log only on failure.
        echo "✗ openssl configure/build failed for $IOS_SLICE_NAME"
        tail -25 "$build/openssl.log" 2>/dev/null || true
        exit 1
    }

    if [ ! -f "$out/lib/libssl.a" ] || [ ! -f "$out/lib/libcrypto.a" ]; then
        echo "✗ openssl build incomplete for $IOS_SLICE_NAME"
        exit 1
    fi

    # Prove the archive is really iOS and not a stray macOS build — a
    # silently-macOS libuv cost an afternoon; not repeating that here.
    # (`lipo -info` would NOT catch it: both are arm64.)
    echo "✓ openssl $IOS_SLICE_NAME → $out"
    ios_assert_platform "$out/lib/libcrypto.a" "libcrypto.a ($IOS_SLICE_NAME)"
    ios_assert_no_macos "$out/lib/libcrypto.a" "libcrypto.a ($IOS_SLICE_NAME)"
    ios_assert_no_macos "$out/lib/libssl.a"    "libssl.a ($IOS_SLICE_NAME)"
}

case "${1:-all}" in
    device) build_one device ;;
    sim)    build_one sim ;;
    all)    build_one device; build_one sim ;;
    *) echo "Usage: $0 [device|sim|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mOpenSSL for iOS done:\033[0m"
ls -lh "$REPO_ROOT"/dist/openssl-ios-*/lib/*.a 2>/dev/null || true
