#!/usr/bin/env bash
# checksums.sh — write dist/SHA256SUMS covering every distributed artefact.
#
# R-34 (AUDIT-2026-10-29): dist/SHA256SUMS listed three library files and none
# of the CLI tarballs, so a downloaded release could not be verified at all.
# Every archive, every loose binary and the public headers are now covered.
#
# Usage: bash scripts/pack/checksums.sh [dist-dir]

set -u

DIST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/dist}"
OUT="$DIST/SHA256SUMS"
TMP="$OUT.tmp"

cd "$DIST" || exit 1

{
    # CLI binaries and archives.
    for f in sofuu-darwin-* sofuu-linux-* sofuu-windows-*; do
        [ -f "$f" ] || continue
        case "$f" in *.sha256) continue ;; esac
        echo "$f"
    done
    # Embeddable libraries, the C header, and the QTSQ staging tree contents.
    for f in libsofuu.a libsofuu.dylib libsofuu-linux-*.a libsofuu-linux-*.so \
             libsofuu-darwin-*.a libsofuu-darwin-*.dylib libsofuu-android-*.so \
             libsofuu-ios-*.a sofuu_embed.h; do
        [ -f "$f" ] && echo "$f"
    done
} | sort -u | while IFS= read -r f; do
    shasum -a 256 "$f" 2>/dev/null || sha256sum "$f" 2>/dev/null
done > "$TMP"

if [ -s "$TMP" ]; then
    mv "$TMP" "$OUT"
    echo "✓ dist/SHA256SUMS: $(wc -l < "$OUT" | tr -d ' ') artefacts"
else
    rm -f "$TMP"
    echo "✗ dist/SHA256SUMS: nothing to hash in $DIST" >&2
    exit 1
fi
