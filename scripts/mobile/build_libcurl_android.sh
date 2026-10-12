#!/usr/bin/env bash
# scripts/mobile/build_libcurl_android.sh — static, TLS-capable libcurl
# per Android ABI (the runtime's HTTP/TLS transport).
#
# The zig cross curl (scripts/cross/build_libcurl_static.sh) is
# deliberately TLS-free — unusable for a phone chat app that talks to
# OpenAI/Anthropic. This build links the OpenSSL built by
# build_openssl_android.sh instead, and pins the same curl 8.6.0 tarball
# with the same hash the cross script verifies.
#
# Produces: dist/libcurl-android-<abi>/lib/libcurl.a (+ include)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ndk_env.sh" "$@"

CURL_VER="8.6.0"
CURL_TARBALL="curl-${CURL_VER}.tar.gz"
CURL_URL="https://curl.se/download/${CURL_TARBALL}"
# Same pin as scripts/cross/build_libcurl_static.sh (P2-24).
CURL_SHA256="9c6db808160015f30f3c656c0dec125feb9dc00753596bf858a272b5dd8dc398"
CURL_SRC="/tmp/curl-${CURL_VER}"

build_one() {
    local abi="$1"
    mobile_abi_env "$abi"
    local out="$REPO_ROOT/dist/libcurl-android-$MOBILE_ABI_NAME"
    local ssl="$REPO_ROOT/dist/openssl-android-$MOBILE_ABI_NAME"
    if [ -f "$out/lib/libcurl.a" ]; then
        echo "✓ curl $MOBILE_ABI_NAME already built"
        return 0
    fi
    [ -f "$ssl/lib/libssl.a" ] || { echo "✗ openssl missing for $MOBILE_ABI_NAME — run build_openssl_android.sh first"; exit 1; }

    if [ ! -x "/tmp/curl-${CURL_VER}/build-android-$MOBILE_ABI_NAME/Makefile" ] && [ ! -f "$CURL_SRC/.verified-sha" ]; then
        echo "→ downloading curl ${CURL_VER}"
        [ -f "/tmp/$CURL_TARBALL" ] || curl -fsSL --max-time 120 -o "/tmp/$CURL_TARBALL" "$CURL_URL"
        local got
        got="$(shasum -a 256 "/tmp/$CURL_TARBALL" | cut -d' ' -f1)"
        [ "$got" = "$CURL_SHA256" ] || { echo "✗ curl tarball checksum mismatch"; exit 1; }
        rm -rf "$CURL_SRC"
        tar xf "/tmp/$CURL_TARBALL" -C /tmp
        printf '%s\n' "$CURL_SHA256" > "$CURL_SRC/.verified-sha"
    fi

    echo "── curl ${CURL_VER} (OpenSSL) → $MOBILE_ABI_NAME"
    local build="$CURL_SRC/build-android-$MOBILE_ABI_NAME"
    mkdir -p "$build" "$out/lib"
    cmake -S "$CURL_SRC" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_INSTALL_PREFIX="$out" \
        -DCMAKE_TOOLCHAIN_FILE="$MOBILE_CMAKE_TOOLCHAIN" \
        -DANDROID_ABI="$MOBILE_ABI_NAME" \
        -DANDROID_PLATFORM="android-$MOBILE_API" \
        -DANDROID_NDK="$MOBILE_NDK_HOME" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
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
        -DCURL_CA_BUNDLE=none \
        -DCURL_CA_PATH=none >/dev/null 2>&1
    cmake --build "$build" >/dev/null 2>&1
    cmake --install "$build" >/dev/null 2>&1
    [ -f "$out/lib/libcurl.a" ] || { echo "✗ curl build incomplete for $MOBILE_ABI_NAME"; exit 1; }
    # The runtime links curl by name (-lstatic=curl), so MERGE the OpenSSL
    # archives into libcurl.a: one -l then carries the whole TLS stack and
    # X509_*/SSL_* resolve inside libsofuu.so at dlopen time on device.
    # (Extract-then-rearchive — `ar q archive.a other.a` appends the nested
    # ARCHIVE as a member, which the linker ignores.)
    local merge="$build/merge"; rm -rf "$merge"; mkdir -p "$merge"
    ( cd "$merge" \
        && "$MOBILE_AR" x "$out/lib/libcurl.a" \
        && mkdir -p ossl && cd ossl \
        && "$MOBILE_AR" x "$ssl/lib/libssl.a" \
        && "$MOBILE_AR" x "$ssl/lib/libcrypto.a" \
        && for f in *.o; do mv "$f" "ossl_$f"; done )
    ( cd "$merge" && "$MOBILE_AR" crs "$out/lib/libcurl.a" $(ls -1 *.o ossl/*.o) )
    echo "✓ curl $MOBILE_ABI_NAME → $out/lib/libcurl.a (OpenSSL merged in)"
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
