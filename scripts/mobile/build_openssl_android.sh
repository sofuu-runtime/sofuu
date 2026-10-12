#!/usr/bin/env bash
# scripts/mobile/build_openssl_android.sh — static OpenSSL per Android ABI.
#
# Android has no system OpenSSL: the runtime's TLS stack is libcurl, and
# libcurl on Android must bring its own TLS backend. This script builds
# libcrypto.a + libssl.a from the official OpenSSL source tarball with the
# NDK clang (no-asm: deterministic, no perlasm dependency; the payloads
# here are small JSON over TLS, so the asm win is noise).
#
# Supply chain: the tarball comes from ftp.openssl.org and is verified
# against the .sha256 published next to it by the same origin (the repo's
# pinned-hash convention, without hardcoding a hash this machine cannot
# independently attest).
#
# Usage: bash scripts/mobile/build_openssl_android.sh [arm64|x86_64 …]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/mobile/ndk_env.sh" "$@"

OPENSSL_VER="3.3.2"
# ftp.openssl.org was retired in 2025 — the official release asset now
# lives on the openssl org's GitHub releases. TLS from the origin is the
# primary check; the pinned sha256 below is that asset's hash, so any
# later tampering (or a different payload) fails the build.
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VER}/openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SHA256="2e8a40b01979afe8be0bbfb3de5dc1c6709fedb46d6c89c10da114ab5fc3d281"
OPENSSL_TARBALL="openssl-${OPENSSL_VER}.tar.gz"
OPENSSL_SRC="/tmp/openssl-${OPENSSL_VER}"

build_one() {
    local abi="$1"
    mobile_abi_env "$abi"
    local out="$REPO_ROOT/dist/openssl-android-$MOBILE_ABI_NAME"
    # Cached under either layout (older builds staged at the root).
    if [ -f "$out/lib/libssl.a" ] || [ -f "$out/libssl.a" ]; then
        mkdir -p "$out/lib"
        [ -f "$out/lib/libssl.a" ] || cp "$out/libssl.a" "$out/libcrypto.a" "$out/lib/" 2>/dev/null || true
        echo "✓ openssl $MOBILE_ABI_NAME already built"
        return 0
    fi
    if [ ! -f "$OPENSSL_SRC/Configure" ]; then
        echo "→ downloading openssl ${OPENSSL_VER} (official release asset)"
        rm -f "/tmp/$OPENSSL_TARBALL"
        curl -fsSL --max-time 180 -o "/tmp/$OPENSSL_TARBALL" "$OPENSSL_URL"
        local got
        got="$(shasum -a 256 "/tmp/$OPENSSL_TARBALL" | cut -d' ' -f1)"
        [ "$got" = "$OPENSSL_SHA256" ] || { echo "✗ openssl tarball checksum mismatch"; exit 1; }
        rm -rf "$OPENSSL_SRC"
        tar xzf "/tmp/$OPENSSL_TARBALL" -C /tmp
    fi
    echo "── openssl ${OPENSSL_VER} → $MOBILE_ABI_NAME ($MOBILE_OPENSSL_TARGET)"
    local build="$OPENSSL_SRC/build-android-$MOBILE_ABI_NAME"
    rm -rf "$build"; mkdir -p "$build"
    (
        cd "$build"
        PERL="${PERL:-perl}" "$OPENSSL_SRC/Configure" \
            "$MOBILE_OPENSSL_TARGET" \
            no-shared no-tests no-module \
            --prefix="$out" --openssldir="$out" \
            >/dev/null
        make -s -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" build_libs >/dev/null
        mkdir -p "$out/lib" "$out/include"
        cp libssl.a libcrypto.a "$out/lib/"
        # Public headers = the checkout's include tree (ossl_typ.h,
        # opensslv.h, the full openssl/*.h set) with the build tree's
        # GENERATED headers (ssl.h from ssl.h.in, opensslconf.h, …)
        # overlaid on top.
        cp -R "$OPENSSL_SRC/include/." "$out/include/"
        cp -R "$build/include/." "$out/include/" 2>/dev/null || true
    )
    [ -f "$out/lib/libssl.a" ] && { [ -f "$out/include/openssl/ssl.h" ] || [ -f "$out/include/ssl.h" ]; } || {
        echo "✗ openssl build incomplete for $MOBILE_ABI_NAME"; exit 1; }
    echo "✓ openssl $MOBILE_ABI_NAME → $out"
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
