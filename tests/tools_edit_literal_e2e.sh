#!/usr/bin/env bash
# tests/tools_edit_literal_e2e.sh — js-6 (AUDIT-2026-09-07) regression.
#
# edit_file single-match mode must splice new_string LITERALLY (the old
# String.replace branch interpreted the replacement-pattern sequences and
# corrupted files). Drives the REAL binary via `sofuu run` against a
# scratch project dir; the driver makes cwd-relative bare-name edits whose
# new_string carries the full set of replacement-pattern sequences, built
# at runtime. The gate is the driver's ALL-PASSED string — `sofuu run`
# does not reliably propagate exit codes.
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
  "$SOFUU" run "$ROOT/tests/tools_edit_literal_test.js" > "$TMP/out.txt" 2>&1
RC=$?

cat "$TMP/out.txt"

if ! grep -q "EDIT-LITERAL TEST: ALL PASSED" "$TMP/out.txt"; then
    echo "TOOLS-EDIT-LITERAL E2E: FAILED (driver gate string missing, rc=$RC)"
    exit 1
fi

# Forensics: the driver's finally-block must have removed every fixture.
if ls "$PROJ"/edit_literal_*.txt >/dev/null 2>&1; then
    echo "TOOLS-EDIT-LITERAL E2E: FAILED (leftover fixtures in scratch project)"
    exit 1
fi

echo "TOOLS-EDIT-LITERAL E2E: ALL PASSED"
