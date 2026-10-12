#!/usr/bin/env bash
# scripts/mobile/build_libcurl_ios.sh — static, TLS-capable libcurl per iOS
# slice (the runtime's HTTP/TLS transport).
#
# The iOS twin of build_libcurl_android.sh: same curl tarball, same pinned
# hash, same protocol trimming, same trick of merging the OpenSSL archives
# into libcurl.a so a single `-lstatic=curl` carries the whole TLS stack.
#
# Two iOS-specific decisions:
#
#  * **zlib is disabled on purpose.** `crates/sofuu-ffi/build.rs` only emits
#    `-lz` inside the QTSQ branch, and the iOS pack is QTSQ-free, so the final
#    app link has no `-lz`. Letting curl find the SDK's zlib would produce an
#    archive with undefined `_inflate` and fail the app link. `CURL_ZLIB` is
#    not an option in curl 8.6 (it uses `optional_dependency(ZLIB)`), so the
#    find is disabled outright.
#  * **SecureTransport is not an option either** — curl 8.6 removed that
#    backend in 8.0, so OpenSSL it is, exactly as on Android.
#
# Produces: dist/libcurl-ios-<slice>/lib/libcurl.a (+ include)
#
# Usage: bash scripts/mobile/build_libcurl_ios.sh [device|sim|all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ios_env.sh"

CURL_VER="8.6.0"
CURL_TARBALL="curl-${CURL_VER}.tar.gz"
CURL_URL="https://curl.se/download/${CURL_TARBALL}"
# Same pin as build_libcurl_android.sh / scripts/cross/build_libcurl_static.sh.
CURL_SHA256="9c6db808160015f30f3c656c0dec125feb9dc00753596bf858a272b5dd8dc398"
CURL_SRC="/tmp/curl-${CURL_VER}"

build_one() {
    local slice="$1"
    ios_slice_env "$slice"

    local out="$REPO_ROOT/dist/libcurl-ios-$IOS_SLICE_NAME"
    local ssl="$REPO_ROOT/dist/openssl-ios-$IOS_SLICE_NAME"

    if [ -f "$out/lib/libcurl.a" ]; then
        echo "✓ curl $IOS_SLICE_NAME already built"
        return 0
    fi
    if [ ! -f "$ssl/lib/libssl.a" ] || [ ! -f "$ssl/lib/libcrypto.a" ]; then
        echo "✗ openssl missing for $IOS_SLICE_NAME — run build_openssl_ios.sh first"
        exit 1
    fi

    if [ ! -f "$CURL_SRC/CMakeLists.txt" ]; then
        echo "→ downloading curl ${CURL_VER}"
        [ -f "/tmp/$CURL_TARBALL" ] || curl -fsSL --max-time 120 -o "/tmp/$CURL_TARBALL" "$CURL_URL"
        local got
        got="$(shasum -a 256 "/tmp/$CURL_TARBALL" | cut -d' ' -f1)"
        [ "$got" = "$CURL_SHA256" ] || {
            echo "✗ curl tarball checksum mismatch"
            echo "  expected $CURL_SHA256"
            echo "  got      $got"
            exit 1
        }
        rm -rf "$CURL_SRC"
        tar xf "/tmp/$CURL_TARBALL" -C /tmp
    fi

    echo "── curl ${CURL_VER} (OpenSSL) → $IOS_SLICE_NAME ($IOS_ARCH)"
    local build="$CURL_SRC/build-ios-$IOS_SLICE_NAME"
    rm -rf "$build"
    mkdir -p "$build" "$out/lib" "$out/include"

    # shellcheck disable=SC2086
    cmake -S "$CURL_SRC" -B "$build" $IOS_CMAKE_GEN \
        $IOS_CMAKE_FLAGS \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_INSTALL_PREFIX="$out" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_TESTING=OFF \
        -DCURL_DISABLE_LDAP=ON \
        -DCURL_DISABLE_LDAPS=ON \
        -DCURL_DISABLE_RTSP=ON \
        -DCURL_DISABLE_POP3=ON \
        -DCURL_DISABLE_IMAP=ON \
        -DCURL_DISABLE_SMTP=ON \
        -DCURL_DISABLE_GOPHER=ON \
        -DCURL_DISABLE_FTP=ON \
        -DCURL_DISABLE_TELNET=ON \
        -DCURL_DISABLE_TFTP=ON \
        -DCURL_USE_LIBSSH2=OFF \
        -DCURL_USE_LIBPSL=OFF \
        -DCURL_USE_OPENSSL=ON \
        -DOPENSSL_ROOT_DIR="$ssl" \
        -DOPENSSL_INCLUDE_DIR="$ssl/include" \
        -DOPENSSL_CRYPTO_LIBRARY="$ssl/lib/libcrypto.a" \
        -DOPENSSL_SSL_LIBRARY="$ssl/lib/libssl.a" \
        -DUSE_NGHTTP2=OFF \
        -DCURL_BROTLI=OFF \
        -DCURL_ZSTD=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON \
        -DCURL_CA_BUNDLE=none \
        -DCURL_CA_PATH=none \
        >/dev/null

    cmake --build "$build" >/dev/null
    cmake --install "$build" >/dev/null

    if [ ! -f "$out/lib/libcurl.a" ]; then
        echo "✗ curl build incomplete for $IOS_SLICE_NAME"
        exit 1
    fi

    # The runtime links curl by name (-lstatic=curl), so MERGE the OpenSSL
    # archives into libcurl.a: one -l then carries the whole TLS stack.
    # (Extract-then-rearchive — `ar q archive.a other.a` appends the nested
    # ARCHIVE as a member, which the linker ignores.) libssl and libcrypto
    # go into separate dirs so same-named members cannot clobber each other.
    local merge="$build/merge"
    rm -rf "$merge"
    mkdir -p "$merge/curl" "$merge/ssl" "$merge/crypto"
    ( cd "$merge/curl"   && "$IOS_AR" x "$out/lib/libcurl.a" )
    ( cd "$merge/ssl"    && "$IOS_AR" x "$ssl/lib/libssl.a" )
    ( cd "$merge/crypto" && "$IOS_AR" x "$ssl/lib/libcrypto.a" )
    (
        cd "$merge"
        for f in ssl/*.o;    do mv "$f" "ssl_$(basename "$f")"; done
        for f in crypto/*.o; do mv "$f" "crypto_$(basename "$f")"; done
        "$IOS_AR" crs "$out/lib/libcurl.a" curl/*.o ssl_*.o crypto_*.o
        "$IOS_RANLIB" "$out/lib/libcurl.a"
    )

    # Prove the archive is iOS, not macOS — the failure that cost an
    # afternoon on libuv, checked here instead of discovered later.
    # `lipo -info` is not enough (both are arm64); vtool reads the actual
    # LC_BUILD_VERSION platform out of a member object.
    echo "✓ curl $IOS_SLICE_NAME → $out/lib/libcurl.a (OpenSSL merged in)"
    ios_assert_platform "$out/lib/libcurl.a" "libcurl.a ($IOS_SLICE_NAME)"
    ios_assert_no_macos "$out/lib/libcurl.a" "libcurl.a ($IOS_SLICE_NAME)"
}

case "${1:-all}" in
    device) build_one device ;;
    sim)    build_one sim ;;
    all)    build_one device; build_one sim ;;
    *) echo "Usage: $0 [device|sim|all]"; exit 1 ;;
esac

echo ""
echo "  \033[32mlibcurl for iOS done:\033[0m"
ls -lh "$REPO_ROOT"/dist/libcurl-ios-*/lib/libcurl.a 2>/dev/null || true
