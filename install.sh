#!/usr/bin/env sh
# Sofuu install script — https://sofuu.dev/install
# Usage: curl -fsSL https://sofuu.dev/install | sh
set -e

REPO="sofuu-runtime/sofuu"
INSTALL_DIR="${SOFUU_INSTALL_DIR:-/usr/local/bin}"
BINARY="sofuu"

# ── Detect platform ───────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    arm64|aarch64) ARCH="arm64"  ;;
    x86_64)        ARCH="x86_64" ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

case "$OS" in
    darwin) PLATFORM="darwin" ;;
    linux)  PLATFORM="linux"  ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

ASSET_NAME="sofuu-${PLATFORM}-${ARCH}"
ARCHIVE_NAME="${ASSET_NAME}.tar.gz"
CHECKSUM_NAME="${ARCHIVE_NAME}.sha256"

# ── Fetch latest version tag ──────────────────────────────────
echo "→ Detecting latest release..."
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//')

if [ -z "$LATEST" ]; then
    echo "Could not determine latest release."
    echo "Visit: https://github.com/${REPO}/releases"
    exit 1
fi

BASE_URL="https://github.com/${REPO}/releases/download/${LATEST}"
ARCHIVE_URL="${BASE_URL}/${ARCHIVE_NAME}"
CHECKSUM_URL="${BASE_URL}/${CHECKSUM_NAME}"

echo "→ Downloading sofuu ${LATEST} (${PLATFORM}/${ARCH})..."
TMP_DIR=$(mktemp -d)
curl -fsSL "$ARCHIVE_URL" -o "${TMP_DIR}/${ARCHIVE_NAME}"

# ── Verify checksum ───────────────────────────────────────────
echo "→ Verifying checksum..."
curl -fsSL "$CHECKSUM_URL" -o "${TMP_DIR}/${CHECKSUM_NAME}"
cd "$TMP_DIR"
# sha256sum on Linux, shasum on macOS
if command -v sha256sum > /dev/null 2>&1; then
    sha256sum -c "${CHECKSUM_NAME}"
elif command -v shasum > /dev/null 2>&1; then
    # Convert format: sha256sum uses "hash  file" but shasum -a 256 expects same
    shasum -a 256 -c "${CHECKSUM_NAME}"
fi
cd -

# ── Extract ───────────────────────────────────────────────────
tar -xzf "${TMP_DIR}/${ARCHIVE_NAME}" -C "$TMP_DIR"
chmod +x "${TMP_DIR}/${ASSET_NAME}"

# ── Install ───────────────────────────────────────────────────
echo "→ Installing to ${INSTALL_DIR}/${BINARY}..."
if [ -w "$INSTALL_DIR" ]; then
    mv "${TMP_DIR}/${ASSET_NAME}" "${INSTALL_DIR}/${BINARY}"
else
    sudo mv "${TMP_DIR}/${ASSET_NAME}" "${INSTALL_DIR}/${BINARY}"
fi

rm -rf "$TMP_DIR"

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "✅ Sofuu ${LATEST} installed successfully!"
echo ""
"${INSTALL_DIR}/${BINARY}" version
echo ""
echo "   Docs: https://sofuu.xyz"
echo "   Run:  sofuu run app.js"
echo "   REPL: sofuu"
echo ""
