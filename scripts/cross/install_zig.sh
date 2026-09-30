#!/usr/bin/env bash
# scripts/cross/install_zig.sh
# Download and install Zig into ./tools/zig (no sudo, no brew)
# Zig is used as a zero-dependency cross-compiler for Linux targets.
set -euo pipefail

ZIG_VERSION="0.13.0"
TOOLS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/tools"
ZIG_DIR="$TOOLS_DIR/zig"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map to Zig arch names
case "$ARCH" in
    arm64|aarch64) ZIG_ARCH="aarch64" ;;
    x86_64)        ZIG_ARCH="x86_64"  ;;
    *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

case "$OS" in
    darwin) ZIG_OS="macos" ;;
    linux)  ZIG_OS="linux" ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

TARBALL="zig-${ZIG_OS}-${ZIG_ARCH}-${ZIG_VERSION}.tar.xz"
URL="https://ziglang.org/download/${ZIG_VERSION}/${TARBALL}"

# Official SHA-256 of each 0.13.0 tarball (ziglang.org/download/index.json).
# P3 (AUDIT-2026-09-07): the download used to be piped straight into tar —
# a compromised mirror or MITM'd fetch would execute as build tooling.
case "${ZIG_OS}-${ZIG_ARCH}" in
    macos-aarch64) EXPECTED_SHA="46fae219656545dfaf4dce12fb4e8685cec5b51d721beee9389ab4194d43394c" ;;
    macos-x86_64)  EXPECTED_SHA="8b06ed1091b2269b700b3b07f8e3be3b833000841bae5aa6a09b1a8b4773effd" ;;
    linux-x86_64)  EXPECTED_SHA="d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea" ;;
    linux-aarch64) EXPECTED_SHA="041ac42323837eb5624068acd8b00cd5777dac4cf91179e8dad7a7e90dd0c556" ;;
    *) echo "No pinned checksum for ${ZIG_OS}-${ZIG_ARCH}; refusing to install unsigned" >&2; exit 1 ;;
esac

sha256_ok() {
    # $1 = file, $2 = expected hex digest
    if command -v sha256sum >/dev/null 2>&1; then
        echo "$2  $1" | sha256sum -c - >/dev/null 2>&1
    else
        echo "$2  $1" | shasum -a 256 -c - >/dev/null 2>&1
    fi
}

if [ -x "$ZIG_DIR/zig" ]; then
    echo "✓ Zig already installed: $($ZIG_DIR/zig version)"
    exit 0
fi

mkdir -p "$TOOLS_DIR"
echo "→ Downloading Zig ${ZIG_VERSION} (${ZIG_OS}/${ZIG_ARCH})..."
curl -fsSL "$URL" -o "/tmp/${TARBALL}"

echo "→ Verifying SHA-256..."
if ! sha256_ok "/tmp/${TARBALL}" "$EXPECTED_SHA"; then
    echo "✗ Checksum mismatch for ${TARBALL} (expected ${EXPECTED_SHA})" >&2
    rm -f "/tmp/${TARBALL}"
    exit 1
fi

echo "→ Extracting..."
tar -xf "/tmp/${TARBALL}" -C "$TOOLS_DIR"
mv "$TOOLS_DIR/zig-${ZIG_OS}-${ZIG_ARCH}-${ZIG_VERSION}" "$ZIG_DIR"
rm "/tmp/${TARBALL}"

echo ""
echo "✅ Zig $($ZIG_DIR/zig version) installed to $ZIG_DIR"
echo "   Add to PATH: export PATH="/opt/homebrew/bin:$PATH"=\"$ZIG_DIR:\$PATH\""
