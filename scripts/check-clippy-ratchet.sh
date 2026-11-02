#!/usr/bin/env bash
# scripts/check-clippy-ratchet.sh — warnings may fall, never rise (PLAN-AUDIT-REMEDIATION §5 Task 3.2).
#
# Reads .ci/clippy-baseline.txt, runs clippy over the shipped crates, and fails
# if any crate emits more warnings than its baseline. --update rewrites the
# baseline from today's numbers (only do that when the number went DOWN).
#
#   make clippy-ratchet
#   scripts/check-clippy-ratchet.sh --update
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/.ci/clippy-baseline.txt"
PKGS="sofuu-core sofuu-ffi sofuu-capi"

UPDATE=0
case "${1:-}" in
    --update) UPDATE=1 ;;
    "") ;;
    *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac

[ -f "$BASE" ] || { echo "clippy-ratchet: missing $BASE" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
cd "$ROOT" || exit 2

echo "clippy-ratchet: running clippy over $PKGS (this takes a minute)"
# shellcheck disable=SC2086
cargo clippy $(for p in $PKGS; do printf -- '-p %s ' "$p"; done) --lib --bins 2>&1 \
    | grep -E 'generated [0-9]+ warning' \
    | sed -E 's/^warning: .([a-z0-9_-]+). \((lib|build script)\) generated ([0-9]+) warning.*/\1:\2:\3/' \
    | sed 's/:build script:/:build:/' \
    | sort -u >"$TMP/now.txt"

if [ ! -s "$TMP/now.txt" ]; then
    echo "clippy-ratchet: clippy produced no warning summary — did it build?" >&2
    exit 2
fi

if [ "$UPDATE" -eq 1 ]; then
    {
        sed -n '1,/^sofuu-core/p' "$BASE" | grep -E '^#' || true
        cat "$TMP/now.txt"
    } >"$TMP/new_base.txt"
    mv "$TMP/new_base.txt" "$BASE"
    echo "clippy-ratchet: baseline updated"
    cat "$TMP/now.txt"
    exit 0
fi

fail=0
while IFS=: read -r crate target now; do
    was=$(sed -nE "s/^$crate:$target:([0-9]+)$/\1/p" "$BASE")
    if [ -z "$was" ]; then
        echo "clippy-ratchet: $crate:$target is not in the baseline ($now warnings today) — add it"
        fail=1
        continue
    fi
    if [ "$now" -gt "$was" ]; then
        echo "clippy-ratchet: FAIL $crate:$target rose $was -> $now"
        fail=1
    elif [ "$now" -lt "$was" ]; then
        echo "clippy-ratchet: $crate:$target fell $was -> $now — lower the baseline"
    else
        echo "clippy-ratchet: $crate:$target $now (at baseline)"
    fi
done <"$TMP/now.txt"

if [ "$fail" -ne 0 ]; then
    echo "clippy-ratchet: WARNINGS ROSE"
    exit 1
fi
echo "clippy-ratchet: OK"
exit 0
