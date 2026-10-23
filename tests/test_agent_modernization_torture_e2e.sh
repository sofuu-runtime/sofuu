#!/usr/bin/env bash
# tests/test_agent_modernization_torture_e2e.sh -- Phase 6 integration gate.
#
# 50 concurrent agent runs over the real binary: sandbox, context, scheduling,
# QTSQ traces and file editing in one swarm. Runs in a scratch project dir
# because edit_file is jailed to cwd and the fixtures must not land in the
# repo root. The gate is the driver's ALL-PASSED string - "sofuu run" does
# not reliably propagate exit codes.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
if [ ! -f "$SOFUU" ]; then
    echo "sofuu binary not found -- run 'make' first" >&2
    exit 1
fi

TMP="$(mktemp -d)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'rm -rf "$TMP"' EXIT

cd "$PROJ"
perl -e 'alarm 300; exec @ARGV' -- \
  env HOME="$HOME_P" SOFUU_PROJECT="$PROJ" \
  SOFUU_QTSQ_DIR="${SOFUU_QTSQ_DIR:-$HOME/projects/black-hole-disk}" \
  "$SOFUU" run "$ROOT/tests/test_agent_modernization_torture.js" > "$TMP/out.txt" 2>&1
RC=$?

cat "$TMP/out.txt"

if ! grep -q "ALL PHASE 6 TORTURE TESTS PASSED" "$TMP/out.txt"; then
    echo "TORTURE E2E: FAILED (gate string missing, rc=$RC)"
    exit 1
fi

if grep -q "  FAIL " "$TMP/out.txt"; then
    echo "TORTURE E2E: FAILED (a check failed)"
    exit 1
fi

if ls "$PROJ"/torture_* >/dev/null 2>&1; then
    echo "TORTURE E2E: FAILED (leftover fixtures in scratch project)"
    exit 1
fi

echo "TORTURE E2E: ALL PASSED"
