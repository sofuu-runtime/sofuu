#!/usr/bin/env bash
# scripts/cross/build_libcurl_static.sh
#
# Fetches and builds a minimal, statically-linkable libcurl using Zig as CC.
# Produces: dist/libcurl-<target>/lib/libcurl.a
#
# This avoids needing a system libcurl on the cross-target.
# Only includes: HTTP/HTTPS + TLS (system ssl is mbedTLS — no OpenSSL dep).
#
# Usage:
#   bash scripts/cross/build_libcurl_static.sh x86_64-linux-musl
#   bash scripts/cross/build_libcurl_static.sh aarch64-linux-musl
#
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

TARGET="${1:?Usage: $0 <zig-target>}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"
CURL_VER="8.6.0"
CURL_TARBALL="curl-${CURL_VER}.tar.gz"
CURL_URL="https://curl.se/download/${CURL_TARBALL}"
CURL_SRC="/tmp/curl-${CURL_VER}"
CURL_INSTALL="$DIST/libcurl-${TARGET}"

# Normalise target for naming  
ARCH=$(echo "$TARGET" | cut -d- -f1)
case "$ARCH" in
    x86_64)   CMAKE_ARCH="x86_64"  ;;
    aarch64)  CMAKE_ARCH="aarch64" ;;
    *)        echo "Unknown arch"; exit 1 ;;
esac

if [ -f "$CURL_INSTALL/lib/libcurl.a" ]; then
    echo "✓ libcurl already built for $TARGET"
    exit 0
fi

if [ ! -x "$ZIG" ]; then
    echo "Zig not found. Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

echo "→ Downloading curl ${CURL_VER}..."
if [ ! -d "$CURL_SRC" ]; then
    curl -fsSL "$CURL_URL" -o "/tmp/${CURL_TARBALL}"
    tar -xf "/tmp/${CURL_TARBALL}" -C /tmp
fi

echo "→ Configuring curl for ${TARGET}..."
BUILD_DIR="${CURL_SRC}/build-${TARGET}"
mkdir -p "$BUILD_DIR" "$CURL_INSTALL"

# Use Zig as C compiler via CMake
cmake -S "$CURL_SRC" -B "$BUILD_DIR" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_INSTALL_PREFIX="$CURL_INSTALL" \
    -DCMAKE_C_COMPILER="$ZIG" \
    -DCMAKE_C_COMPILER_ARG1="cc -target ${TARGET}" \
    -DCMAKE_C_FLAGS="-target ${TARGET}" \
    -DCMAKE_EXE_LINKER_FLAGS="-target ${TARGET} -static" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR="$CMAKE_ARCH" \
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
    -DCURL_USE_LIBSSL=OFF \
    -DCURL_USE_MBEDTLS=OFF \
    -DCURL_USE_OPENSSL=OFF \
    -DCURL_USE_GNUTLS=OFF \
    -DCURL_USE_BEARSSL=OFF \
    -DCMAKE_USE_OPENSSL=OFF \
    -DUSE_NGHTTP2=OFF \
    -DCURL_CA_BUNDLE=none \
    -DCURL_CA_PATH=none \
    2>&1 | tail -5

echo "→ Building curl..."
cmake --build "$BUILD_DIR" -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5
cmake --install "$BUILD_DIR"

echo ""
echo "✅ Built: $CURL_INSTALL/lib/libcurl.a"
echo ""
