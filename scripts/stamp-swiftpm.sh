#!/usr/bin/env bash
# scripts/stamp-swiftpm.sh — pin the xcframework checksum in Package.swift.
#
# SwiftPM binary targets REQUIRE a `checksum:` and refuse to resolve without
# a correct one, so it cannot be left as a placeholder in a published package.
# This script recomputes the SHA-256 of a built xcframework zip and rewrites
# the constant in Package.swift, then verifies the file it wrote.
#
# Usage:
#   SOFUU_VERSION=0.2.0 scripts/stamp-swiftpm.sh dist/libsofuu.xcframework.zip
#
# Run after `make dist-ios` and after attaching the artifact to a release.
# If SOFUU_VERSION is set, the download URL is rewritten to match too.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${1:-}"

if [[ -z "$ZIP" ]]; then
  echo "usage: SOFUU_VERSION=<v> $0 <libsofuu.xcframework.zip>" >&2
  exit 2
fi
if [[ ! -f "$ZIP" ]]; then
  echo "error: no such file: $ZIP" >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  SHA="$(shasum -a 256 "$ZIP" | awk '{print $1}')"
else
  SHA="$(sha256sum "$ZIP" | awk '{print $1}')"
fi

VERSION="${SOFUU_VERSION:-}"

python3 - "$REPO_ROOT/Package.swift" "$SHA" "$VERSION" <<'PY'
import re, sys
path, sha, version = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()

# Checksum: always rewritten.
new, n = re.subn(r'(let sofuuXCFrameworkChecksum\s*=\s*")[0-9a-fA-F]{64}(")',
                 lambda m: m.group(1) + sha + m.group(2), src)
if n != 1:
    sys.exit(f"error: expected exactly 1 checksum constant, found {n}")
src = new

# URL: only when a version was supplied.
if version:
    v = version.lstrip("v")
    url = f"https://github.com/sofuu-runtime/sofuu/releases/download/v{v}/libsofuu.xcframework.zip"
    new, n = re.subn(r'(let sofuuXCFrameworkURL\s*=\s*\n?\s*)"[^"]*"',
                     lambda m: m.group(1) + '"' + url + '"', src)
    if n != 1:
        sys.exit(f"error: expected exactly 1 url constant, found {n}")
    src = new

open(path, "w").write(src)
PY

# Verify what we actually wrote, rather than trusting the rewrite.
if grep -q "let sofuuXCFrameworkChecksum = \"$SHA\"" "$REPO_ROOT/Package.swift"; then
  if grep -qE 'let sofuuXCFrameworkChecksum = "0{64}"' "$REPO_ROOT/Package.swift"; then
    echo "error: checksum still looks like a placeholder after rewrite" >&2
    exit 1
  fi
  echo "✓ Package.swift pinned to sha256:$SHA"
  [[ -n "$VERSION" ]] && echo "  url → v${VERSION#v}"
  exit 0
fi

echo "error: rewrite did not take effect (expected sha256:$SHA)" >&2
exit 1
