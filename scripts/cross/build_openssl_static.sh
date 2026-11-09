#!/usr/bin/env bash
# scripts/cross/build_openssl_static.sh
#
# Fetches and cross-compiles static OpenSSL (libssl.a + libcrypto.a)
# for Linux targets using Zig as C compiler.
#
# Produces: dist/openssl-<target>/lib/libssl.a
#           dist/openssl-<target>/lib/libcrypto.a
#           dist/openssl-<target>/include/
#
# Usage:
#   bash scripts/cross/build_openssl_static.sh x86_64-linux-musl
#   bash scripts/cross/build_openssl_static.sh aarch64-linux-musl
#
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

TARGET="${1:?Usage: $0 <zig-target>}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIG="$REPO_ROOT/tools/zig/zig"
DIST="$REPO_ROOT/dist"

OPENSSL_VER="3.3.2"
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VER}/openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SHA256="2e8a40b01979afe8be0bbfb3de5dc1c6709fedb46d6c89c10da114ab5fc3d281"
OPENSSL_TARBALL="openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SRC="/tmp/openssl-${OPENSSL_VER}"
OPENSSL_INSTALL="$DIST/openssl-${TARGET}"

ARCH=$(echo "$TARGET" | cut -d- -f1)
case "$ARCH" in
    x86_64)   OPENSSL_ARCH="linux-x86_64" ;;
    aarch64)  OPENSSL_ARCH="linux-aarch64" ;;
    *)        echo "Unknown arch: $ARCH"; exit 1 ;;
esac

if [ -f "$OPENSSL_INSTALL/lib/libssl.a" ] && [ -f "$OPENSSL_INSTALL/lib/libcrypto.a" ]; then
    echo "✓ openssl already built for $TARGET"
    exit 0
fi

if [ ! -x "$ZIG" ]; then
    echo "Zig not found. Run: bash scripts/cross/install_zig.sh"
    exit 1
fi

echo "→ Downloading OpenSSL ${OPENSSL_VER}..."
if [ ! -f "/tmp/${OPENSSL_TARBALL}" ]; then
    curl -fsSL --max-time 180 "$OPENSSL_URL" -o "/tmp/${OPENSSL_TARBALL}"
fi

GOT_SHA="$(shasum -a 256 "/tmp/${OPENSSL_TARBALL}" | cut -d' ' -f1)"
if [ "$GOT_SHA" != "$OPENSSL_SHA256" ]; then
    echo "✗ OpenSSL tarball checksum mismatch:"
    echo "    expected $OPENSSL_SHA256"
    echo "    got      $GOT_SHA"
    exit 1
fi

OPENSSL_MARK="${OPENSSL_SRC}/.verified-sha"
if [ ! -d "$OPENSSL_SRC" ] || [ ! -f "$OPENSSL_MARK" ] || [ "$(cat "$OPENSSL_MARK")" != "$OPENSSL_SHA256" ]; then
    rm -rf "$OPENSSL_SRC"
    tar -xzf "/tmp/${OPENSSL_TARBALL}" -C /tmp
    printf '%s\n' "$OPENSSL_SHA256" > "$OPENSSL_MARK"
fi

echo "→ Configuring OpenSSL for ${TARGET} (${OPENSSL_ARCH})..."
BUILD_DIR="${OPENSSL_SRC}/build-${TARGET}"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$OPENSSL_INSTALL/lib" "$OPENSSL_INSTALL/include"

(
    cd "$BUILD_DIR"
    export CC="$ZIG cc -target ${TARGET}"
    export AR="$ZIG ar"
    export RANLIB="$ZIG ranlib"
    PERL="${PERL:-perl}" "$OPENSSL_SRC/Configure" \
        "$OPENSSL_ARCH" \
        no-shared no-tests no-module no-dso no-comp no-engine no-async no-asm \
        --prefix="$OPENSSL_INSTALL" \
        --openssldir="$OPENSSL_INSTALL" \
        >/dev/null

    echo "→ Compiling OpenSSL static libraries..."
    make -s -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" build_libs >/dev/null

    cp libssl.a libcrypto.a "$OPENSSL_INSTALL/lib/"
    cp -R "$OPENSSL_SRC/include/." "$OPENSSL_INSTALL/include/"
    cp -R "$BUILD_DIR/include/." "$OPENSSL_INSTALL/include/" 2>/dev/null || true
)

[ -f "$OPENSSL_INSTALL/lib/libssl.a" ] && [ -f "$OPENSSL_INSTALL/lib/libcrypto.a" ] || {
    echo "✗ OpenSSL build incomplete for $TARGET"; exit 1;
}

echo ""
echo "✅ Built: $OPENSSL_INSTALL/lib/libssl.a + libcrypto.a"
echo ""
