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

# The release asset name. CI publishes libsofuu.xcframework.zip, but v0.3.0-beta
# was published as libsofuu-xcframework.zip - so the name is a variable here, and
# --from-release discovers the real one instead of assuming it.
ASSET_NAME="${SOFUU_XCF_ASSET:-libsofuu.xcframework.zip}"
REPO_SLUG="${SOFUU_REPO_SLUG:-sofuu-runtime/sofuu}"

# --consistency: the two manifests must at least agree with each other.
#
# --check answers "is this tree publishable for vX", which needs SOFUU_VERSION and
# a release in progress. This mode needs neither, so `make test` can run it on
# every commit: it catches the drift the audit found without waiting for a tag.
# The manifests track the most recent *published* release, not Cargo.toml, so
# agreeing with each other is the invariant - the stamp moves both to the new
# version in one pass, and a half-finished stamp (one file rewritten, the other
# left behind) is how Sofuu.podspec stayed on 0.2.0 through two releases.
if [[ "$ZIP" == "--consistency" ]]; then
  fail=0
  pkg_version="$(grep -A1 'let sofuuXCFrameworkURL' "$REPO_ROOT/Package.swift" \
                 | tr -d ' \n' | sed -n 's#.*/download/v\([^/]*\)/libsofuu.xcframework.zip.*#\1#p')"
  pod_version="$(awk -F"'" '/s\.version/ {print $2; exit}' "$REPO_ROOT/Sofuu.podspec")"
  if [[ -z "$pkg_version" ]]; then
    echo "error: could not read a release version out of Package.swift URL." >&2
    fail=1
  elif [[ "$pkg_version" != "$pod_version" ]]; then
    echo "error: the install paths disagree - Package.swift points at v$pkg_version," >&2
    echo "       Sofuu.podspec says s.version = '$pod_version'." >&2
    echo "       A half-stamped release sends SwiftPM and CocoaPods to different artifacts." >&2
    echo "       Fix: SOFUU_VERSION=<v> scripts/stamp-swiftpm.sh dist/libsofuu.xcframework.zip" >&2
    fail=1
  fi
  if grep -qE 'let sofuuXCFrameworkChecksum = "0{64}"' "$REPO_ROOT/Package.swift"; then
    echo "note: Package.swift carries the all-zero checksum placeholder." >&2
    echo "      That is the pre-release state, and README says so; the release stamp" >&2
    echo "      replaces it with the checksum of the published zip." >&2
  fi
  [[ "$fail" -eq 0 ]] && echo "OK SwiftPM and CocoaPods agree: both point at v$pkg_version"
  exit "$fail"
fi
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

# --from-release <tag>: stamp from the artifact that is actually published.
#
# Stamping from a local zip assumes the release will publish those exact bytes,
# and two builds of this xcframework zip differently (73cd6386... vs 8e685239...
# on consecutive builds of the same tree). The published artifact is the only thing
# a consumer can resolve, so this mode reads the release over the API and pins what
# is there - including the asset NAME, which v0.3.0-beta shipped as
# libsofuu-xcframework.zip while every script here produces
# libsofuu.xcframework.zip. A checksum pinned to a 404 resolves nothing.
RELEASE_TAG=""
RELEASE_SHA=""
case "$ZIP" in
  --from-release)
    RELEASE_TAG="${2:-}"
    if [[ -z "$RELEASE_TAG" ]]; then
      echo "usage: $0 --from-release <tag>" >&2
      exit 2
    fi
    meta_file="$(mktemp)"
    trap 'rm -f "$meta_file"' EXIT
    curl -fsSL --max-time 60 \
      "https://api.github.com/repos/$REPO_SLUG/releases/tags/$RELEASE_TAG" > "$meta_file" || {
      echo "error: could not read release $RELEASE_TAG from $REPO_SLUG" >&2; exit 1; }
    python3 - "$meta_file" > "$meta_file.env" <<'PY'
import json, sys, shlex
rel = json.load(open(sys.argv[1]))
tag = rel.get("tag_name") or ""
assets = [a for a in rel.get("assets", []) if a.get("name", "").endswith("xcframework.zip")]
if not tag or not assets:
    sys.exit("error: release has no *xcframework.zip asset")
a = assets[0]
digest = (a.get("digest") or "").split(":")[-1]
if len(digest) != 64:
    sys.exit("error: asset " + a["name"] + " has no sha256 digest from the API")
print("RELEASE_TAG=" + shlex.quote(tag))
print("RELEASE_ASSET=" + shlex.quote(a["name"]))
print("RELEASE_SHA=" + shlex.quote(digest))
print("RELEASE_SIZE=" + shlex.quote(str(a.get("size", 0))))
PY
    if [[ $? -ne 0 ]]; then
      echo "error: could not resolve a signed artifact in $RELEASE_TAG" >&2
      exit 1
    fi
    . "$meta_file.env"
    ASSET_NAME="$RELEASE_ASSET"
    echo "release $RELEASE_TAG: $RELEASE_ASSET ($RELEASE_SIZE B, sha256 ${RELEASE_SHA:0:12}...)"
    ;;
esac
if [[ -z "$ZIP" && -z "$RELEASE_TAG" ]]; then
  echo "usage: SOFUU_VERSION=<v> $0 <libsofuu.xcframework.zip>" >&2
  echo "       SOFUU_VERSION=<v> $0 --check" >&2
  exit 2
fi
if [[ -z "$RELEASE_TAG" && ! -f "$ZIP" ]]; then
  echo "error: no such file: $ZIP" >&2
  exit 1
fi

if [[ -n "$RELEASE_TAG" ]]; then
  SHA="$RELEASE_SHA"
elif command -v shasum >/dev/null 2>&1; then
  SHA="$(shasum -a 256 "$ZIP" | awk '{print $1}')"
else
  SHA="$(sha256sum "$ZIP" | awk '{print $1}')"
fi

# The API reports the tag with its leading v; the manifests carry the bare
# version and the URL template adds the v back. Stripping it here is what keeps
# `pod 'Sofuu', '~> 0.3'` resolvable - without it the podspec said "v0.3.0-beta"
# and the asset URL read "vv0.3.0-beta".
VERSION="${SOFUU_VERSION:-${RELEASE_TAG:-}}"
VERSION="${VERSION#v}"

python3 - "$REPO_ROOT/Package.swift" "$SHA" "$VERSION" "$ASSET_NAME" "$REPO_SLUG" <<'PY'
import re, sys
path, sha, version, asset, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
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
    url = "https://github.com/%s/releases/download/v%s/%s" % (repo, v, asset)
    new, n = re.subn(r'(let sofuuXCFrameworkURL\s*=\s*\n?\s*)"[^"]*"',
                     lambda m: m.group(1) + '"' + url + '"', src)
    if n != 1:
        sys.exit(f"error: expected exactly 1 url constant, found {n}")
    src = new

open(path, "w").write(src)
PY

# The podspec carries the same version + URL pair and drifts on its own, so it
# stamped in the same pass. A podspec rewrite that fails is fatal: an artifact
# pinned by SwiftPM but not by CocoaPods is the drift this script exists to stop.
#
# This block used $V, which is only assigned in the --check branch above, so the
# stamping path died with "V: unbound variable" under set -u *after* rewriting
# Package.swift and before touching the podspec - leaving the SwiftPM checksum
# updated and CocoaPods still on v0.2.0. The version for this path is $VERSION.
if [[ -n "$VERSION" && -f "$REPO_ROOT/Sofuu.podspec" ]]; then
  python3 - "$REPO_ROOT/Sofuu.podspec" "$VERSION" "$ASSET_NAME" "$REPO_SLUG" <<'PY'
import re, sys
path, version, asset, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
src = open(path).read()
src, n1 = re.subn(r"(s\.version\s*=\s*')[^']*(')",
                  lambda m: m.group(1) + version + m.group(2), src, count=1)
url = "https://github.com/%s/releases/download/v%s/%s" % (repo, version, asset)
src, n2 = re.subn(r"(:http\s*=>\s*')[^']*(')",
                  lambda m: m.group(1) + url + m.group(2), src, count=1)
if n1 != 1 or n2 != 1:
    sys.exit(f"error: podspec rewrite matched version={n1} url={n2} (expected 1 and 1)")
open(path, "w").write(src)
PY
  grep -q "s.version          = '$VERSION'" "$REPO_ROOT/Sofuu.podspec" ||
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
