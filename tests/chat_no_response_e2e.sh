#!/usr/bin/env bash
# tests/chat_no_response_e2e.sh — what does the chat show when the provider
# fails? Runs the REAL chat binary against tests/mock_fail_server.js in six
# failure modes and prints exactly what the user would see.
# SOFUU_STALL_TIMEOUT_SECS=3 shrinks the 5-minute unresponsive-model patience
# window so the MODESILENT turn aborts in ~4s instead of ~5min.
# SOFUU_CHAT_MAX_STEPS=2 shrinks the chat step budget so the MODELOOP turn
# breaches after two tool rounds — the chat must print the salvage summary
# plus a "⏹ stopped: step budget (2 rounds) reached" line, never a bare
# "(no response)".
# MODECLOSE must be the LAST mode turn: it kills the mock process (zero
# response bytes → curl transport error, e.g. "Empty reply from server").
# The retry classifier must treat that as transient: the chat retries twice
# against the now-dead port ("Couldn't connect to server", ~5.5s of backoff)
# and fails loudly — the final "Couldn't connect" proves attempt 1 was
# classified transient and retried instead of surfacing immediately.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-18790}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

TMP="$(mktemp -d /tmp/sofuu_noresp_XXXXXX)"
HOME_DIR="$TMP/home"
mkdir -p "$HOME_DIR/.sofuu"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

"$SOFUU" run "$ROOT/tests/mock_fail_server.js" "$PORT" >"$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 50); do grep -q "MOCK-FAIL-READY" "$TMP/mock.log" 2>/dev/null && break; sleep 0.1; done

printf 'MODE429 hello\nMODEERRFRAME hello\nMODEEMPTY hello\nMODESILENT hello\nMODELOOP hello\nMODECLOSE hello\n/exit\n' |
  HOME="$HOME_DIR" SOFUU_STALL_TIMEOUT_SECS=3 SOFUU_CHAT_MAX_STEPS=2 "$SOFUU" chat 2>&1 | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'

echo "===== mock server log ====="
cat "$TMP/mock.log"
echo "===== config after session (no_think_models?) ====="
python3 -c "import json;print(json.load(open('$HOME_DIR/.sofuu/config.json')).get('no_think_models'))" 2>/dev/null || cat "$HOME_DIR/.sofuu/config.json"
