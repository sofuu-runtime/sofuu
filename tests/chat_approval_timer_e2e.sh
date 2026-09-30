#!/usr/bin/env bash
# tests/chat_approval_timer_e2e.sh — js-3 (AUDIT-2026-09-07): the approval
# hard-ceiling timer leaks on early resolve. Each gated tool call arms a
# toolTimeoutMs (desktop: 1h) setTimeout that is never cleared when the
# approval resolves; rt/timer.rs caps the registry at MAX_TIMERS = 1024
# and setTimeout THROWS past it. 140 rounds x 8 gated write_file calls =
# 1120 approvals forces the overflow on the unfixed driver: approval
# promises start rejecting, write_file never runs, "tool error: too many
# timers" rows appear, files go missing. Post-fix the timers are cleared
# on resolve and all 1120 files land.
#
# Drives the JS engine (sofuu.chat.submit) via ./sofuu run, cwd = project
# so the write_file jail root is the throwaway project dir. Isolated HOME,
# mock LLM in-process (the driver owns the mock server), no network, no
# real keys. Exits non-zero on any failure.
# Run:  bash tests/chat_approval_timer_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18931 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_ap_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "active": "mock-ap",
  "providers": [ { "name": "mock-ap", "model": "mock-model",
                   "endpoint": "$BASE", "api_key": "x" } ],
  "brain": false, "ghost": false, "sync": false }
EOF

OUT="$TMP/driver.log"
perl -e 'alarm 240; exec @ARGV' -- \
  env HOME="$HOME_P" SOFUU_PROJECT="$PROJ" SOFUU_QTSQ_DIR="${SOFUU_QTSQ_DIR:-$HOME/projects/black-hole-disk}" \
    "$SOFUU" run "$ROOT/tests/chat_approval_timer_driver.js" "$PORT" > "$OUT" 2>&1
RC=$?
grep -q "MOCK-AP-READY" "$OUT" || { echo "FAIL mock server never became ready"; cat "$OUT"; exit 1; }

if [ $RC -eq 0 ]; then echo "PASS driver exit 0"; else echo "FAIL driver exit 0 (rc=$RC)"; fi

for s in "exactly one done event" "all 1120 approvals requested and resolved ok" \
         "zero tool errors (approval path never threw)" "all 1120 files written" \
         "done.answer = APPROVAL-TIMER-OK"; do
  if grep -q "PASS $s" "$OUT"; then echo "PASS $s"; else echo "FAIL $s"; FAILURES=$((FAILURES+1)); fi
done

# Forensics on failure: first tool-error samples + mock round log tail.
if [ $FAILURES -gt 0 ]; then
  echo "--- driver log (head) ---"; head -40 "$OUT"
  echo "--- driver log (tail) ---"; tail -25 "$OUT"
fi

echo "APPROVAL-TIMER E2E: $FAILURES failures"
exit $([ $FAILURES -eq 0 ] && echo 0 || echo 1)
