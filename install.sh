#!/usr/bin/env sh
# Sofuu install script — https://sofuu.xyz/install
# Usage: curl -fsSL https://sofuu.xyz/install | sh
#
# Prefer a package manager? The same binary installs via:
#   npm install -g sofuu      (or: npx sofuu)
#   https://sofuu.xyz/downloads  (tarball + .sha256, any platform)
set -e

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
    linux)
        # The project ships the macOS CLI only, for now, so no Linux archive
        # is published. Say that plainly instead of 404-ing on a download the
        # user cannot act on. The build path is untouched — `make
        # linux-x86_64` still cross-compiles — so this branch goes away with
        # the release job, not with any capability.
        cat >&2 <<EOF
  Sofuu currently ships a macOS build only.

  Host: ${OS}/${ARCH}. No Linux archive is published right now, so there is
  nothing to download. Re-run this script on macOS, or build from source:

      git clone https://github.com/sofuu-runtime/sofuu
      cd sofuu && SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> make
EOF
        exit 1
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

ASSET_NAME="sofuu-${PLATFORM}-${ARCH}"
ARCHIVE_NAME="${ASSET_NAME}.tar.gz"
CHECKSUM_NAME="${ARCHIVE_NAME}.sha256"

# ── Download location (served straight from sofuu.xyz) ──────
BASE_URL="https://sofuu.xyz/downloads"
ARCHIVE_URL="${BASE_URL}/${ARCHIVE_NAME}"
CHECKSUM_URL="${BASE_URL}/${CHECKSUM_NAME}"

echo "→ Downloading sofuu (${PLATFORM}/${ARCH})..."
TMP_DIR=$(mktemp -d)
if ! curl -fsSL "$ARCHIVE_URL" -o "${TMP_DIR}/${ARCHIVE_NAME}"; then
    echo "✗ No pre-built binary for ${PLATFORM}/${ARCH} yet." >&2
    echo "  Published builds live at https://sofuu.xyz/downloads" >&2
    exit 1
fi

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
else
    echo "✗ No checksum tool (sha256sum/shasum) found — refusing to install an unverified binary." >&2
    exit 1
fi
cd -

# ── Sanity-check archive members ──────────────────────────────
# P3 (AUDIT-2026-09-07): extraction used to run on whatever the tarball
# contained. A compromised or MITM'd archive could carry `../` or
# absolute-path members that escape the temp dir while extracting. The
# checksum above already ties the tarball to the release; this refuses
# any member that would write outside $TMP_DIR regardless.
if tar -tzf "${TMP_DIR}/${ARCHIVE_NAME}" | grep -Eq '(^|/)\.\.(/|$)|^/'; then
    echo "✗ Archive contains path-traversal members — refusing to extract." >&2
    exit 1
fi

# ── Extract ───────────────────────────────────────────────────
tar -xzf "${TMP_DIR}/${ARCHIVE_NAME}" -C "$TMP_DIR"
chmod +x "${TMP_DIR}/${ASSET_NAME}"

# P3 (AUDIT-2026-09-07): the checksum above covers the tarball only. If the
# checksums file also lists the raw binary, verify the extracted file
# directly — a packing mistake (or a tampered extraction) is caught here.
# When the release ships only the archive's hash, this is a no-op.
BIN_LINE=$(grep -E "[[:space:]]\*?${ASSET_NAME}\$" "${TMP_DIR}/${CHECKSUM_NAME}" 2>/dev/null | head -1 || true)
if [ -n "$BIN_LINE" ]; then
    EXPECTED_HASH=$(printf '%s\n' "$BIN_LINE" | cut -d' ' -f1)
    if command -v sha256sum > /dev/null 2>&1; then
        ACTUAL_HASH=$(sha256sum "${TMP_DIR}/${ASSET_NAME}" | cut -d' ' -f1)
    else
        ACTUAL_HASH=$(shasum -a 256 "${TMP_DIR}/${ASSET_NAME}" | cut -d' ' -f1)
    fi
    if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
        echo "✗ Extracted binary checksum mismatch — refusing to install." >&2
        exit 1
    fi
    echo "→ Extracted binary checksum verified."
fi

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
echo "✅ Sofuu installed successfully!"
echo ""
"${INSTALL_DIR}/${BINARY}" version
echo ""
echo "   Docs: https://sofuu.xyz"
echo "   Run:  sofuu run app.js"
echo "   REPL: sofuu"
echo ""
