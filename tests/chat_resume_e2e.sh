#!/usr/bin/env bash
# tests/chat_resume_e2e.sh — E2E for structured session resume (P1):
# a tool-using turn's activity is persisted with the session (.qtsq) and
# replayed into the model's context on resume — the resumed model knows
# which files were read and what came back, instead of inheriting only
# prompt/answer prose.
#
# No network, no API keys (the mock endpoint ignores auth). Exits
# non-zero on any failure.
# Run:  bash tests/chat_resume_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18711 + RANDOM % 79))}"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_resume_XXXXXX)"
HOME_R="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_R/.sofuu" "$PROJ"
if [ "${RESUME_E2E_KEEP:-0}" = "1" ]; then trap 'kill $MOCK_PID 2>/dev/null' EXIT;
else trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT; fi

printf 'RESUME-TOOL-CANARY-9271\nline two\n' > "$PROJ/notes.txt"

cat > "$HOME_R/.sofuu/config.json" <<EOF
{ "provider": "openai", "model": "mock-resume", "base_url": "http://127.0.0.1:$PORT/v1/chat/completions",
  "api_key": "x", "ml": true, "brain": true,
  "providers": [ { "name": "openai", "model": "mock-resume",
                   "endpoint": "http://127.0.0.1:$PORT/v1/chat/completions",
                   "api_key": "x" } ],
  "active": "openai" }
EOF

./sofuu run tests/mock_resume_server.js "$PORT" > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 50); do
  grep -q "MOCK-RESUME-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done

RESUME_PROJ="$PROJ" HOME="$HOME_R" \
  "$SOFUU" run tests/chat_resume_driver.js "$PORT" > "$TMP/driver.log" 2>&1
DRIVER_RC=$?

sed 's/^/  driver: /' "$TMP/driver.log" | tail -12
check "resume driver exit 0" "$DRIVER_RC"
grep -q "resume context reached the model" "$TMP/mock.log"
check "mock saw the resume context" $?
echo "  --- mock.log (request shapes) ---"
sed 's/^/  mock: /' "$TMP/mock.log" | tail -8

# ── Project-store layout assertions (P: everything under .sofuu/) ──
PROJ_STORE="$PROJ/.sofuu"
test -d "$PROJ_STORE/brain"; check "brain is project-local (brain/ folder)" $?
test -f "$PROJ_STORE/brain/index.qtsq"; check "brain manifest (brain/index.qtsq)" $?
test -f "$PROJ_STORE/sessions/registry.qtsq"; check "qtsq-only registry (registry.qtsq)" $?
SID_DIR=$(ls -d "$PROJ_STORE/sessions"/s-* 2>/dev/null | head -1)
test -n "$SID_DIR" && test -f "$SID_DIR/session.qtsq"; check "session folder with session.qtsq index" $?
CONV_COUNT=$(ls "$SID_DIR"/conv/*.qtsq 2>/dev/null | wc -l | tr -d ' ')
[ "${CONV_COUNT:-0}" -ge 6 ]; check "real-time conversation: $CONV_COUNT per-event qtsq files (>= 6)" $?
test -f "$PROJ_STORE/debug/.keep" -a -f "$PROJ_STORE/issues/.keep" -a -f "$PROJ_STORE/audits/.keep"
check "debug/ + issues/ + audits/ folders generated" $?
! ls "$PROJ_STORE/sessions/registry.json" >/dev/null 2>&1; check "no plaintext registry written for new sessions" $?

echo
if [ "$FAILURES" -eq 0 ]; then echo "RESUME E2E: ALL PASSED"; else
  echo "RESUME E2E: $FAILURES FAILURE(S)"; exit 1; fi
