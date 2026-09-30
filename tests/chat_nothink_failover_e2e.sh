#!/usr/bin/env bash
# tests/chat_nothink_failover_e2e.sh — js-2 (AUDIT-2026-09-07): the chat
# noThink retry must re-enter the failover classification when IT fails,
# instead of escaping the turn with no 'done' event.
#
# Vehicle: the JS chat engine (src/js/chat.js — the driver the desktop
# runs), exercised in-process via sofuu.chat.submit in
# tests/chat_nothink_failover_driver.js. The TUI (`sofuu chat`) is the
# WRONG vehicle for this finding — it runs the Rust chat.rs driver, which
# has its own noThink handling and never reaches this JS code path.
#
# Scenario (all against one in-process mock, discriminated on the wire):
#   ask 1  model=mock-active, reasoning_effort PRESENT → HTTP 400
#          "reasoning_effort is not supported" → chat.js noThink branch
#   retry  model=mock-active, reasoning_effort ABSENT → HTTP 429
#          (capacity class — this is the throw js-2 is about)
#   fix    429 falls into the failover loop → warn "failing over to
#          custom2" → ask 3 model=mock-failover → plain SSE answer
#          NOTHINK-FAILOVER-OK, servedBy = mock-failover, one 'done'.
#   bug    the 429 escapes the loop: submit() rejects, no failover warn,
#          no done event.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18840 + RANDOM % 120))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_nothink_XXXXXX)"
HOME_DIR="$TMP/home"
mkdir -p "$HOME_DIR/.sofuu" "$TMP/proj"
trap 'rm -rf "$TMP"' EXIT

# The active provider's MODEL is "o3" (a registry reasoning family):
# ai.rs strips reasoning_effort on the wire for models unknown to its
# registry, so an invented name could never reproduce the 400 the
# noThink branch reacts to. The provider NAME stays mock-active.
cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "active": "mock-active",
  "providers": [
    { "name": "mock-active", "model": "o3", "endpoint": "$BASE", "api_key": "x" },
    { "name": "custom2", "model": "mock-failover", "endpoint": "$BASE", "api_key": "x" }
  ],
  "brain": false, "ghost": false, "effort": "high" }
EOF

# The driver binds the mock AND runs the chat engine in one process.
perl -e 'alarm 120; exec @ARGV' -- \
  env HOME="$HOME_DIR" SOFUU_PROJECT="$TMP/proj" SOFUU_STREAM_RETRY_DELAYS="300,300" \
  "$SOFUU" run "$ROOT/tests/chat_nothink_failover_driver.js" "$PORT" >"$TMP/out" 2>&1
RC=$?
check "driver exited 0 (all in-driver checks passed)" "$RC"
grep -q "MOCK-NOTHINK: active served 400 reasoning_effort" "$TMP/out"
check "ask 1 hit the mock's 400 reasoning_effort path" $?
grep -q "MOCK-NOTHINK: active served 429 on the noThink retry" "$TMP/out"
check "the effort-less retry hit the mock's 429 capacity path" $?
grep -q "MOCK-NOTHINK: served plain ok (model=mock-failover)" "$TMP/out"
check "the failover provider served the final answer" $?

echo ""
echo "===== driver + mock output ====="
cat "$TMP/out"

if [ "$FAILURES" -eq 0 ]; then
  echo ""
  echo "CHAT NOTHINK-FAILOVER E2E: ALL PASSED"
  exit 0
fi
echo ""
echo "CHAT NOTHINK-FAILOVER E2E: $FAILURES FAILURE(S) — output above"
exit 1
