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

if [ -x "$ZIG_DIR/zig" ]; then
    echo "✓ Zig already installed: $($ZIG_DIR/zig version)"
    exit 0
fi

mkdir -p "$TOOLS_DIR"
echo "→ Downloading Zig ${ZIG_VERSION} (${ZIG_OS}/${ZIG_ARCH})..."
curl -fsSL "$URL" -o "/tmp/${TARBALL}"

echo "→ Extracting..."
tar -xf "/tmp/${TARBALL}" -C "$TOOLS_DIR"
mv "$TOOLS_DIR/zig-${ZIG_OS}-${ZIG_ARCH}-${ZIG_VERSION}" "$ZIG_DIR"
rm "/tmp/${TARBALL}"

echo ""
echo "✅ Zig $($ZIG_DIR/zig version) installed to $ZIG_DIR"
echo "   Add to PATH: export PATH="/opt/homebrew/bin:$PATH"=\"$ZIG_DIR:\$PATH\""
