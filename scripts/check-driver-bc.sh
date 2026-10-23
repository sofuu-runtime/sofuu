#!/usr/bin/env bash
# scripts/check-driver-bc.sh - build-time guard for the shipped driver
# snapshots (PLAN-AUTONOMOUS-AGENT-LOOP-AND-DRIVERS Phase 6).
#
# Why this exists: a release build embeds src/js/<driver>.bc and no source
# fallback, and the loader prefers the snapshot and returns on first success.
# A snapshot that lags its source would therefore ship stale code with no
# runtime signal at all. This recompiles every driver and fails on any byte
# difference.
#
# Byte comparison is sound because QuickJS bytecode is deterministic for a
# given binary (verified: recompiling twice yields identical bytes), and the
# snapshots in the tree were produced by the same driver set the binary
# embeds. A snapshot made by an OLDER binary can differ in atom indices, so
# after a driver edit the fix is always: scripts/gen-driver-bc.sh && make.
#
#   make bc-check
#   scripts/check-driver-bc.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
if [ ! -f "$SOFUU" ]; then
    echo "sofuu binary not found - run 'make' first" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# compile-bc embeds the input path in the bytecode, so the paths here must
# match the ones scripts/gen-driver-bc.sh used: relative to the repo root.
cd "$ROOT" || exit 1

fail=0
n=0
for js in src/js/*.js; do
    bc="${js%.js}.bc"
    name="$(basename "$bc")"
    if [ ! -f "$bc" ]; then
        echo "MISSING  $name - run scripts/gen-driver-bc.sh"
        fail=1
        continue
    fi
    if ! "$SOFUU" compile-bc "$js" "$TMP/$name" >/dev/null 2>&1; then
        echo "COMPILE  $js failed to compile"
        fail=1
        continue
    fi
    n=$((n + 1))
    if ! cmp -s "$bc" "$TMP/$name"; then
        echo "STALE    $name: the snapshot does not match its source"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "BC-CHECK: FAILED - run scripts/gen-driver-bc.sh and rebuild"
    exit 1
fi
echo "BC-CHECK: OK ($n shipped snapshots match their source)"
