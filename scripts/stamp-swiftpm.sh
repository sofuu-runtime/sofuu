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

# --check: assert the checked-in manifest is publishable for SOFUU_VERSION,
# without an artifact in hand. A release uses this to prove the tree it is
# tagging actually resolves; the stamping path below only proves it could.
if [[ "$ZIP" == "--check" ]]; then
  V="${SOFUU_VERSION:-}"
  if [[ -z "$V" ]]; then
    echo "usage: SOFUU_VERSION=<v> $0 --check" >&2
    exit 2
  fi
  V="${V#v}"
  fail=0
  if grep -qE 'let sofuuXCFrameworkChecksum = "0{64}"' "$REPO_ROOT/Package.swift"; then
    echo "error: Package.swift still carries the all-zero checksum placeholder." >&2
    echo "       SwiftPM cannot resolve a binary target without a real checksum." >&2
    echo "       Stamp it after building the artifact, then commit the result:" >&2
    echo "         SOFUU_VERSION=$V scripts/stamp-swiftpm.sh dist/libsofuu.xcframework.zip" >&2
    fail=1
  fi
  if ! grep -q "/v$V/" "$REPO_ROOT/Package.swift"; then
    echo "error: Package.swift does not point at a v$V release asset." >&2
    echo "       Found: $(grep -A1 'let sofuuXCFrameworkURL' "$REPO_ROOT/Package.swift" | tail -1 | tr -d ' ')" >&2
    fail=1
  fi

  # The podspec is the second install path and drifts independently: it sat at
  # 0.2.0 while the project was on 0.3.0-beta, so "pod 'Sofuu', '~> 0.2'" fetched
  # a release four months older than the one being tagged.
  PODSPEC="$REPO_ROOT/Sofuu.podspec"
  if [[ ! -f "$PODSPEC" ]]; then
    echo "error: Sofuu.podspec is missing." >&2
    fail=1
  else
    # awk -F"'" rather than sed: BSD sed has no \s, and the greedy form silently
    # returned the whole assignment line as the "version".
    pod_version="$(awk -F"'" '/s\.version/ {print $2; exit}' "$PODSPEC")"
    if [[ "$pod_version" != "$V" ]]; then
      echo "error: Sofuu.podspec says s.version = '$pod_version', the release is '$V'." >&2
      echo "       A pod version that does not match the tag resolves to an older artifact." >&2
      fail=1
    fi
    if ! grep -q "/v$V/" "$PODSPEC"; then
      echo "error: Sofuu.podspec does not point at a v$V release asset." >&2
      fail=1
    fi
  fi

  [[ "$fail" -eq 0 ]] && echo "✓ Package.swift and Sofuu.podspec are publishable for v$V"
  exit "$fail"
fi

if [[ -z "$ZIP" ]]; then
  echo "usage: SOFUU_VERSION=<v> $0 <libsofuu.xcframework.zip>" >&2
  echo "       SOFUU_VERSION=<v> $0 --check" >&2
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

# The podspec carries the same version + URL pair and drifts on its own, so it is
# stamped in the same pass. A podspec rewrite that fails is fatal: an artifact
# pinned by SwiftPM but not by CocoaPods is the drift this script exists to stop.
if [[ -n "$V" && -f "$REPO_ROOT/Sofuu.podspec" ]]; then
  python3 - "$REPO_ROOT/Sofuu.podspec" "$V" <<'PY'
import re, sys
path, version = sys.argv[1], sys.argv[2]
src = open(path).read()
src, n1 = re.subn(r"(s\.version\s*=\s*')[^']*(')",
                  lambda m: m.group(1) + version + m.group(2), src, count=1)
url = f"https://github.com/sofuu-runtime/sofuu/releases/download/v{version}/libsofuu.xcframework.zip"
src, n2 = re.subn(r"(:http\s*=>\s*')[^']*(')",
                  lambda m: m.group(1) + url + m.group(2), src, count=1)
if n1 != 1 or n2 != 1:
    sys.exit(f"error: podspec rewrite matched version={n1} url={n2} (expected 1 and 1)")
open(path, "w").write(src)
PY
  grep -q "s.version          = '$V'" "$REPO_ROOT/Sofuu.podspec" ||
    { echo "error: podspec version did not take effect" >&2; exit 1; }
fi

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
