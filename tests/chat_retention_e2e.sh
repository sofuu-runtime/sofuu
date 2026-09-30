#!/usr/bin/env bash
# tests/chat_retention_e2e.sh — E2E for Claude-style transcript
# retention (2026-09-03 e): a tool turn's transcript (assistant
# tool_calls + tool results) PERSISTS into conversation history —
#   1. agent.run returns res.transcript (shape + snapshot safety)
#   2. the chat driver's history holds user → assistant+tc → tool →
#      assistant with no orphaned tool message
#   3. a plain follow-up turn's outbound request still carries the
#      previous turn's tool result (the mock answers the needle marker
#      only when it sees it)
#   4. /compact folds whole blocks — the history stays orphan-free
#
# No network, no API keys (the mock endpoint ignores auth). Exits
# non-zero on any failure.
# Run:  bash tests/chat_retention_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18731 + RANDOM % 59))}"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_retention_XXXXXX)"
HOME_R="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_R/.sofuu" "$PROJ"
if [ "${RETENTION_E2E_KEEP:-0}" = "1" ]; then trap 'kill $MOCK_PID 2>/dev/null' EXIT;
else trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT; fi

printf 'NEEDLE-TT-88 notes line one\nnotes line two\n' > "$PROJ/notes.txt"

cat > "$HOME_R/.sofuu/config.json" <<EOF
{ "provider": "openai", "model": "mock-retention",
  "base_url": "http://127.0.0.1:$PORT/v1/chat/completions",
  "api_key": "x", "ml": false, "brain": false, "rlm": "off" }
EOF

"$SOFUU" run tests/mock_retention_server.js "$PORT" > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 50); do
  grep -q "MOCK-RET-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-RET-READY" "$TMP/mock.log" || { echo "FAIL mock server did not start"; exit 1; }

RETENTION_PROJ="$PROJ" HOME="$HOME_R" \
  "$SOFUU" run tests/chat_retention_driver.js "$PORT" > "$TMP/driver.log" 2>&1
DRIVER_RC=$?

sed 's/^/  driver: /' "$TMP/driver.log"
check "retention driver exit 0" "$DRIVER_RC"
grep -q "PASS L2 plain turn 2 request carried the retained transcript" "$TMP/driver.log"
check "driver saw the retained transcript cross turns" $?

# Mock forensics: the plain turn ("now just summarize it") request must
# have arrived WITH a tool-role message — retention on the wire, not just
# in the driver's in-memory history.
PLAIN_REQ_SAW_TOOL=0
while IFS= read -r line; do
  if printf '%s' "$line" | grep -q 'last="now just summarize it"'; then
    if printf '%s' "$line" | grep -q 'toolmsg=yes'; then
      PLAIN_REQ_SAW_TOOL=1
    fi
  fi
done < "$TMP/mock.log"
[ "$PLAIN_REQ_SAW_TOOL" -eq 1 ]
check "mock saw tool messages in the plain turn's request" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT RETENTION E2E: ALL PASSED"
  exit 0
else
  echo "CHAT RETENTION E2E: $FAILURES FAILURES"
  exit 1
fi
