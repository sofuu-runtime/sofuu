#!/usr/bin/env bash
# stage_legal.sh — put the licence files into a release staging directory.
#
# H-06 (AUDIT-2026-10-29): v0.3.0-beta shipped release tarballs and npm
# packages that contained the bare binary and *nothing else*. MIT obliges a
# redistributor to reproduce the copyright notice and permission text with any
# binary distribution, so every archive now carries LICENSE + NOTICE.
#
# Usage:
#   bash scripts/pack/stage_legal.sh <dest-dir> [<repo-root>]
#
# Never fails the build when an optional file is missing; LICENSE and NOTICE
# are the two that must exist, and their absence is reported on stderr.

set -u

dest="${1:?usage: stage_legal.sh <dest-dir> [<repo-root>]}"
root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

mkdir -p "$dest" || exit 1

missing=0
for f in LICENSE NOTICE; do
    if [ -f "$root/$f" ]; then
        cp "$root/$f" "$dest/" || exit 1
    else
        echo "stage_legal: ✗ $f not found under $root" >&2
        missing=1
    fi
done

# Optional: the README explains the open-core boundary quoted in NOTICE.
[ -f "$root/README.md" ] && cp "$root/README.md" "$dest/"

# Per-component licence texts referenced by NOTICE (QTSQ format, fonts, …).
if [ -d "$root/LICENSES" ]; then
    rm -rf "$dest/LICENSES"
    cp -R "$root/LICENSES" "$dest/" || exit 1
fi

# SHIP-MANIFEST lets a support engineer map an artefact back to its contents
# without unpacking it (R-34: released binaries carried no build stamp at all).
{
    echo "sofuu release artefact"
    echo "staged: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source-commit: $(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "source-dirty: $(git -C "$root" status --porcelain 2>/dev/null | wc -l | tr -d ' ') modified paths"
    echo "contents:"
    (cd "$dest" && ls -1)
} > "$dest/SHIP-MANIFEST.txt"

exit "$missing"
