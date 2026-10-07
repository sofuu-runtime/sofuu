#!/usr/bin/env bash
# tests/chat_empty_session_e2e.sh — unspoken sessions are not saved.
#
# Regression: opening sofuu and closing it without sending anything left
# a saved (empty) session behind — every accidental open polluted the
# /sessions and /resume pickers. finish() on a turn-less session now
# removes its registry entry and store instead of ending it.
#
# Run A (empty): /exit with no turns → registry stays empty, no store.
# Run B (one mock turn + /exit): the session is kept, marked ended.
# Registry assertions are plaintext JSON — no QTSQ needed either way.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29900 + RANDOM % 500))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_empty_session_XXXXXX)"
HOME_DIR="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_DIR/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; wait $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

( cd "$ROOT" && exec env HOME="$HOME_DIR" "$SOFUU" run tests/mock_llm_server.js "$PORT" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-LLM-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$TMP/mock.log" || { echo "mock LLM failed to start"; cat "$TMP/mock.log"; exit 1; }

REG="$PROJ/.sofuu/sessions/registry.json"
STORES="$PROJ/.sofuu/sessions"

# ── run A: open, say nothing, quit ─────────────────────────────
printf '/exit\n' |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/run_a" 2>&1
python3 - "$REG" <<'PYEOF'
import json, sys
reg = json.load(open(sys.argv[1])) if __import__("os").path.exists(sys.argv[1]) else {"sessions": []}
print("sessions:", len(reg.get("sessions", [])))
sys.exit(0 if len(reg.get("sessions", [])) == 0 else 1)
PYEOF
check "empty session leaves no registry entry" $?
[ "$(find "$STORES" -maxdepth 1 -name '*.store' 2>/dev/null | wc -l)" -eq 0 ]
check "empty session leaves no store dir" $?

# ── run B: one turn, then quit ──────────────────────────────────
printf 'hello there\n/exit\n' |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/run_b" 2>&1
grep -q "PLAIN-OK" "$TMP/run_b"
check "control turn completed" $?
python3 - "$REG" <<'PYEOF'
import json, sys
reg = json.load(open(sys.argv[1]))
ss = reg.get("sessions", [])
print("sessions:", len(ss), "ended:", [s.get("ended") for s in ss])
sys.exit(0 if len(ss) == 1 and ss[0].get("ended") is True else 1)
PYEOF
check "spoken session kept exactly once, marked ended" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT EMPTY SESSION E2E: ALL PASSED"
  exit 0
fi
echo "CHAT EMPTY SESSION E2E: $FAILURES FAILURE(S)"
echo "===== run A ====="; cat "$TMP/run_a"
echo "===== run B ====="; cat "$TMP/run_b"
echo "===== registry ====="; cat "$REG" 2>/dev/null
exit 1
