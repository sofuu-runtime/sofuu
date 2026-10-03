#!/usr/bin/env bash
# tests/tools_diff_e2e.sh — TUI edit/write diff regression.
#
# edit_file/write_file must stash a bounded unified diff for the TUI while
# leaving their return strings byte-identical. Drives the REAL binary via
# `sofuu run` against a scratch project dir; the driver makes
# cwd-relative bare-name edits and asserts on the stashed diff. The gate
# is the driver's ALL-PASSED string — `sofuu run` does not reliably
# propagate exit codes.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
if [ ! -f "$SOFUU" ]; then
    echo "✗ sofuu binary not found — run 'make' first" >&2
    exit 1
fi

TMP="$(mktemp -d)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'rm -rf "$TMP"' EXIT

# Isolated HOME + scratch project so the driver's edits never land in the
# repo (a plain TESTS entry would inherit the repo-root cwd and write its
# fixtures there).
cd "$PROJ"
perl -e 'alarm 240; exec @ARGV' -- \
  env HOME="$HOME_P" SOFUU_PROJECT="$PROJ" \
  SOFUU_QTSQ_DIR="${SOFUU_QTSQ_DIR:-$HOME/projects/black-hole-disk}" \
  "$SOFUU" run "$ROOT/tests/tools_diff_test.js" > "$TMP/out.txt" 2>&1
RC=$?

cat "$TMP/out.txt"

if ! grep -q "TOOLS-DIFF TEST: ALL PASSED" "$TMP/out.txt"; then
    echo "TOOLS-DIFF E2E: FAILED (driver gate string missing, rc=$RC)"
    exit 1
fi

# Forensics: the driver's finally-block must have removed every fixture.
if ls "$PROJ"/diff_*.txt >/dev/null 2>&1; then
    echo "TOOLS-DIFF E2E: FAILED (leftover fixtures in scratch project)"
    exit 1
fi

echo "TOOLS-DIFF E2E: ALL PASSED"
