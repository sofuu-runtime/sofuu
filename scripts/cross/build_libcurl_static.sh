#!/usr/bin/env bash
# scripts/cross/build_libcurl_static.sh
#
# Fetches and builds a statically-linkable, TLS-capable libcurl using OpenSSL and Zig as CC.
# Produces: dist/libcurl-<target>/lib/libcurl.a (with OpenSSL merged in)
#
# This provides self-contained HTTPS support for cross-compiled Linux targets.
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
# P2-24 (AUDIT-2026-09-01): pin the tarball — an unverified download is a
# supply-chain hole (zlib was already hash-checked in build_qtsq_cross.sh;
# curl wasn't). Hash verified against the curl.se PGP signature
# (27EDEAF22F3ABCEB50DB9A125CC908FDB71E12C2, Daniel Stenberg).
CURL_SHA256="9c6db808160015f30f3c656c0dec125feb9dc00753596bf858a272b5dd8dc398"
CURL_SRC="/tmp/curl-${CURL_VER}"
CURL_INSTALL="$DIST/libcurl-${TARGET}"
BUILD_DIR="${CURL_SRC}/build-${TARGET}"

# Normalise target for naming  
ARCH=$(echo "$TARGET" | cut -d- -f1)
case "$ARCH" in
    x86_64)   CMAKE_ARCH="x86_64"  ;;
    aarch64)  CMAKE_ARCH="aarch64" ;;
    *)        echo "Unknown arch"; exit 1 ;;
esac

if [ -f "$CURL_INSTALL/lib/libcurl.a" ]; then
    if nm "$CURL_INSTALL/lib/libcurl.a" 2>/dev/null | grep "SSL_connect" >/dev/null; then
        echo "✓ libcurl (with OpenSSL) already built for $TARGET"
        exit 0
    else
        echo "→ Existing libcurl for $TARGET is TLS-free — rebuilding with OpenSSL..."
        rm -rf "$CURL_INSTALL" "$BUILD_DIR"
    fi
fi

if [ ! -x "$ZIG" ]; then
    echo "Zig not found. Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

# ── Ensure static OpenSSL is prebuilt for target ─────────────────
echo "→ Ensuring OpenSSL static is built for ${TARGET}..."
bash "$REPO_ROOT/scripts/cross/build_openssl_static.sh" "$TARGET"
OPENSSL_DIR="$DIST/openssl-${TARGET}"
[ -f "$OPENSSL_DIR/lib/libssl.a" ] || { echo "✗ Missing OpenSSL for $TARGET at $OPENSSL_DIR"; exit 1; }

echo "→ Downloading curl ${CURL_VER}..."
if [ ! -f "/tmp/${CURL_TARBALL}" ]; then
    curl -fsSL "$CURL_URL" -o "/tmp/${CURL_TARBALL}"
fi
# Verify before extracting — cached or fresh, the hash must match the pin.
GOT_SHA="$(shasum -a 256 "/tmp/${CURL_TARBALL}" | cut -d' ' -f1)"
if [ "$GOT_SHA" != "$CURL_SHA256" ]; then
    echo "✗ curl tarball checksum mismatch:"
    echo "    expected $CURL_SHA256"
    echo "    got      $GOT_SHA"
    echo "  Refusing to build from an unverified download."
    exit 1
fi
# P1-17 companion: reuse the extracted tree only when it provably came from
# the verified tarball (marker names the hash); a stale or tampered /tmp
# tree is wiped and re-extracted.
CURL_MARK="${CURL_SRC}/.verified-sha"
if [ ! -d "$CURL_SRC" ] || [ ! -f "$CURL_MARK" ] || [ "$(cat "$CURL_MARK")" != "$CURL_SHA256" ]; then
    rm -rf "$CURL_SRC"
    tar -xf "/tmp/${CURL_TARBALL}" -C /tmp
    printf '%s\n' "$CURL_SHA256" > "$CURL_MARK"
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
    -DCMAKE_C_FLAGS="-target ${TARGET} -fno-asynchronous-unwind-tables -fno-unwind-tables" \
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
    -DCURL_USE_OPENSSL=ON \
    -DOPENSSL_ROOT_DIR="$OPENSSL_DIR" \
    -DOPENSSL_INCLUDE_DIR="$OPENSSL_DIR/include" \
    -DOPENSSL_CRYPTO_LIBRARY="$OPENSSL_DIR/lib/libcrypto.a" \
    -DOPENSSL_SSL_LIBRARY="$OPENSSL_DIR/lib/libssl.a" \
    -DCURL_USE_GNUTLS=OFF \
    -DCURL_USE_BEARSSL=OFF \
    -DCURL_USE_LIBSSH2=OFF \
    -DCURL_USE_LIBPSL=OFF \
    -DUSE_NGHTTP2=OFF \
    -DCURL_CA_BUNDLE=none \
    -DCURL_CA_PATH=none \
    2>&1 | tail -5

echo "→ Building curl..."
cmake --build "$BUILD_DIR" -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5
cmake --install "$BUILD_DIR"

echo "→ Merging OpenSSL static archives into libcurl.a..."
MERGE_DIR="${BUILD_DIR}/merge"
rm -rf "$MERGE_DIR"
mkdir -p "$MERGE_DIR/ossl"
(
    cd "$MERGE_DIR"
    "$ZIG" ar x "$CURL_INSTALL/lib/libcurl.a"
    (
        cd ossl
        "$ZIG" ar x "$OPENSSL_DIR/lib/libssl.a"
        "$ZIG" ar x "$OPENSSL_DIR/lib/libcrypto.a"
        for f in *.o; do mv "$f" "ossl_$f"; done
    )
    find . -name "*.o" | sort > objlist.txt
    rm -f "$CURL_INSTALL/lib/libcurl.a"
    "$ZIG" ar crs "$CURL_INSTALL/lib/libcurl.a" @objlist.txt
    "$ZIG" ranlib "$CURL_INSTALL/lib/libcurl.a" 2>/dev/null || true
)

nm "$CURL_INSTALL/lib/libcurl.a" 2>/dev/null | grep "SSL_connect" >/dev/null || {
    echo "✗ Verification failed: libcurl.a does not contain OpenSSL symbols!"; exit 1;
}

echo ""
echo "✅ Built: $CURL_INSTALL/lib/libcurl.a (with OpenSSL TLS support)"
echo ""
