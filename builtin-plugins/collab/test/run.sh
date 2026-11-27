#!/usr/bin/env bash
# builtin-plugins/collab/test/run.sh — acceptance tests for the collab plugin.
#
# Two suites, both against a REAL runtime, neither involving a model:
#   1. collab_test.js  — claim/conflict/overlap/rollback/TTL/renew/release,
#                        board seq + cursor, turn injection, write gate,
#                        auto file_change, turn-end cleanup, strict mode,
#                        runtime ownership injection.
#   2. the race        — N separate OS processes claim the same path at once;
#                        exactly one may be granted. This is the property a
#                        process-local mutex cannot provide.
#
# Everything runs in a throwaway workspace containing a copy of the plugin, so
# the repository's own .sofuu state is never touched.
#
#   builtin-plugins/collab/test/run.sh
#   SOFUU_BIN=/path/to/sofuu builtin-plugins/collab/test/run.sh
#   KEEP=1 builtin-plugins/collab/test/run.sh      # keep the temp workspace
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${SOFUU_BIN:-$ROOT/target/debug/sofuu}"
[ -x "$BIN" ] || BIN="$ROOT/sofuu"
if [ ! -x "$BIN" ]; then
    echo "✗ no sofuu binary — run 'make' or set SOFUU_BIN" >&2
    exit 1
fi

TMP="$(mktemp -d /tmp/sofuu_collab_XXXXXX)"
cleanup() { [ "${KEEP:-0}" = "1" ] || rm -rf "$TMP"; }
trap cleanup EXIT

mkdir -p "$TMP/plugins"
cp "$ROOT/builtin-plugins/collab/collab.ts" "$TMP/plugins/collab.ts"
cp "$ROOT/builtin-plugins/collab/test/collab_test.js" "$TMP/collab_test.js"
cp "$ROOT/builtin-plugins/collab/test/collab_race_worker.js" "$TMP/race_worker.js"

echo "=== collab unit/contract suite ==="
if ! ( cd "$TMP" && "$BIN" run ./collab_test.js ); then
    echo "✗ collab_test.js failed (workspace: $TMP)" >&2
    KEEP=1
    exit 1
fi

echo
echo "=== cross-process claim race (8 processes) ==="
mkdir -p "$TMP/results"
pids=""
for i in 1 2 3 4 5 6 7 8; do
    ( cd "$TMP" && COLLAB_RACE_ID="$i" "$BIN" run ./race_worker.js >"$TMP/results/$i.log" 2>&1 ) &
    pids="$pids $!"
done
workers_failed=0
for p in $pids; do wait "$p" || workers_failed=$((workers_failed + 1)); done
if [ "$workers_failed" -ne 0 ]; then
    echo "FAIL race workers crashed ($workers_failed)" >&2
    for f in "$TMP"/results/*.log; do [ -s "$f" ] && { echo "--- $f"; cat "$f"; }; done
    KEEP=1
    exit 1
fi

granted=0
for f in "$TMP"/results/*.json; do
    if grep -q '"granted":true' "$f"; then granted=$((granted + 1)); fi
done
total=$(ls "$TMP"/results/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$total" -ne 8 ]; then
    echo "FAIL expected 8 verdicts, found $total" >&2
    KEEP=1
    exit 1
fi
if [ "$granted" -ne 1 ]; then
    echo "FAIL exactly one contender must win, got $granted" >&2
    for f in "$TMP"/results/*.json; do echo "--- $f"; cat "$f"; echo; done
    KEEP=1
    exit 1
fi
echo "PASS exactly one of 8 separate processes won the lease"

# The persisted document must also show one owner; a winner that only exists
# in a process-local table would pass the verdicts and fail here.
owners=$(grep -o '"owner"' "$TMP/.sofuu/collab/leases.json" | wc -l | tr -d ' ')
if [ "$owners" -ne 1 ]; then
    echo "FAIL leases.json contains $owners owner records (expected 1)" >&2
    cat "$TMP/.sofuu/collab/leases.json" >&2
    KEEP=1
    exit 1
fi
echo "PASS leases.json persists exactly one owner"

echo
echo "collab tests: OK"
