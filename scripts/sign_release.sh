#!/usr/bin/env sh
# scripts/sign_release.sh — sign (or verify) a release manifest with minisign.
#
# R7.P4: SHA-256 digests prove the download was not corrupted; a signature
# proves the digest came from us. This script is the signing half; install.sh
# is the verifying half.
#
# Signing (CI, or a release manager's machine):
#   SOFUU_MINISIGN_SECRET_KEY=/path/to/sofuu.key \
#   SOFUU_MINISIGN_PUBKEY=RWQ... \
#     scripts/sign_release.sh dist/SHA256SUMS
#
#   → dist/SHA256SUMS.minisig
#
#   SOFUU_MINISIGN_SECRET_KEY must be a FILE (0600) holding the secret key —
#   never an inline secret in the environment, so it cannot leak through
#   `ps` or a crash dump.
#   SOFUU_MINISIGN_PUBKEY, when set, makes the script verify its own output:
#   a signature that does not validate against the published key is a bug,
#   not a release.
#
# Verifying (anyone):
#   scripts/sign_release.sh --verify dist/SHA256SUMS RWQ...
set -eu

usage() {
    echo "Usage: SOFUU_MINISIGN_SECRET_KEY=<keyfile> [SOFUU_MINISIGN_PUBKEY=<base64>] $0 <manifest>" >&2
    echo "       $0 --verify <manifest> <base64-public-key>" >&2
    exit 2
}

if [ "${1:-}" = "--verify" ]; then
    [ $# -eq 3 ] || usage
    MANIFEST="$2"
    PUBKEY="$3"
    [ -f "$MANIFEST" ] || { echo "✗ manifest not found: $MANIFEST" >&2; exit 1; }
    [ -f "${MANIFEST}.minisig" ] || { echo "✗ signature not found: ${MANIFEST}.minisig" >&2; exit 1; }
    command -v minisign > /dev/null 2>&1 || { echo "✗ minisign is not installed" >&2; exit 1; }
    minisign -Vm "$MANIFEST" -P "$PUBKEY" -x "${MANIFEST}.minisig"
    echo "✓ signature verifies against the published key"
    exit 0
fi

[ $# -eq 1 ] || usage
MANIFEST="$1"
[ -f "$MANIFEST" ] || { echo "✗ manifest not found: $MANIFEST" >&2; exit 1; }
command -v minisign > /dev/null 2>&1 || {
    echo "✗ minisign is not installed (brew install minisign / apt install minisign)" >&2
    exit 1
}

SECRET="${SOFUU_MINISIGN_SECRET_KEY:-}"
if [ -z "$SECRET" ]; then
    echo "SOFUU_MINISIGN_SECRET_KEY is not set — nothing signed." >&2
    echo "  (CI treats this as 'no signing key configured', not as a failure.)" >&2
    exit 3
fi
[ -f "$SECRET" ] || { echo "✗ SOFUU_MINISIGN_SECRET_KEY must be a key FILE (0600), got: $SECRET" >&2; exit 1; }

# Sign with the key file only; minisign may still prompt for its password when
# the key is encrypted, which is exactly what a CI secret store should supply
# through a tty-less agent. -t attaches a trusted comment (accidentally omitted
# in the past; it is part of the signed payload).
minisign -Sm "$MANIFEST" -s "$SECRET" -x "${MANIFEST}.minisig" -t "sofuu release manifest"

PUBKEY="${SOFUU_MINISIGN_PUBKEY:-}"
if [ -n "$PUBKEY" ]; then
    minisign -Vm "$MANIFEST" -P "$PUBKEY" -x "${MANIFEST}.minisig" \
        || { echo "✗ the signature just produced does not verify against SOFUU_MINISIGN_PUBKEY" >&2; exit 1; }
    echo "✓ ${MANIFEST}.minisig verifies against SOFUU_MINISIGN_PUBKEY"
else
    echo "✓ wrote ${MANIFEST}.minisig"
    echo "  (set SOFUU_MINISIGN_PUBKEY to self-verify; publish that key in install.sh)"
fi
