#!/usr/bin/env bash
# tests/chat_state_copy_e2e.sh — js-4 (AUDIT-2026-09-07): state({history:true})
# hands out live tool_calls references. The documented contract is "Copies,
# never live references", but the per-message map passed m.tool_calls through
# BY REFERENCE — a host mutating the returned array silently rewrites the
# live S.history entries that the NEXT turn sends to the provider.
#
# Vehicle: the JS chat engine (src/js/chat.js — the driver the desktop
# runs), exercised in-process via sofuu.chat.submit in
# tests/chat_state_copy_driver.js (same pattern as chat_nothink_failover_e2e).
#
# Scenario (one in-process mock, branched on request CONTENT — a single
# config provider serves both turns):
#   turn 1  write_file tool_call (STATE-CANARY-ORIGINAL into canary.txt),
#           auto-passed via permissionProfile 'full'; the post-tool round
#           trip is answered plainly so the turn settles.
#   driver  reads state({history:true}), finds the tool_calls entry and
#           vandalizes the RETURNED copy (name → read_file, forged args,
#           content → VANDALIZED) exactly as a careless host would.
#   turn 2  the mock asserts server-side (Mimosa-XSS-safe) that the
#           retained history still carries name=write_file + arguments
#           containing STATE-CANARY-ORIGINAL, replying with one of two
#           constant sentences.
#   leak    turn 2 sees the vandalized pair → STATE-CANARY-CORRUPTED.
#   fix     the deep copy shields the engine → STATE-CANARY-INTACT.
# Run:  bash tests/chat_state_copy_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18961 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_sc_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "active": "mock-sc",
  "providers": [ { "name": "mock-sc", "model": "mock-model",
                   "endpoint": "$BASE", "api_key": "x" } ],
  "brain": false, "ghost": false, "sync": false }
EOF

OUT="$TMP/driver.log"
perl -e 'alarm 240; exec @ARGV' -- \
  env HOME="$HOME_P" SOFUU_PROJECT="$PROJ" SOFUU_QTSQ_DIR="${SOFUU_QTSQ_DIR:-$HOME/projects/black-hole-disk}" \
    "$SOFUU" run "$ROOT/tests/chat_state_copy_driver.js" "$PORT" > "$OUT" 2>&1
RC=$?
grep -q "MOCK-STATE-READY" "$OUT" || { echo "FAIL mock server never became ready"; cat "$OUT"; exit 1; }

check "driver exited 0 (all in-driver checks passed)" "$RC"

for s in "turn 1 done" "state({history:true}) exposed a tool_calls entry" \
         "turn 2 done" "turn 2 answer reports intact engine history" \
         "engine history survived the state() mutation (no corruption seen by mock)"; do
  if grep -q "PASS $s" "$OUT"; then echo "PASS $s"; else echo "FAIL $s"; FAILURES=$((FAILURES+1)); fi
done

# Forensics on failure: the mock's turn-2 observation + driver tail.
if [ $FAILURES -gt 0 ]; then
  echo "--- driver log (head) ---"; head -30 "$OUT"
  echo "--- driver log (tail) ---"; tail -20 "$OUT"
fi

echo "STATE-COPY E2E: $FAILURES failures"
exit $([ $FAILURES -eq 0 ] && echo 0 || echo 1)
