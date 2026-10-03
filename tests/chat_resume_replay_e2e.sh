#!/usr/bin/env bash
# tests/chat_resume_replay_e2e.sh — resuming must SHOW the previous chat.
#
# Regression: /resume loaded history silently — the screen showed only a
# "⏺ resumed" notice, so the user stared at an empty conversation with no
# way to see what was resumed. The replay must print the previous turns
# (the same array that entered context, so display and context agree).
#
# Phase 1: one mock turn in a fixture project, then grab that session's
# short id from the sync line. Phase 2: a fresh chat runs /resume <short>
# and must print the old prompt, the old answer, and the resumed notice.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18890 + RANDOM % 200))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_resume_replay_XXXXXX)"
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

# ── phase 1: one real turn, so the session owns a prompt + answer ──
printf 'resume-replay-probe-7z9\n/exit\n' |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/phase1" 2>&1
sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/phase1" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/phase1"
grep -q "PLAIN-OK" "$TMP/phase1"
check "phase 1: mock turn completed (session owns a prompt + answer)" $?
SHORT="$(grep -o 'session s-[0-9a-f]* registered' "$TMP/phase1" | head -1 | awk '{print $2}')"
[ -n "$SHORT" ]
check "phase 1: short session id captured ($SHORT)" $?

# ── phase 2: fresh chat resumes it ───────────────────────────────
printf '/resume %s\n/exit\n' "$SHORT" |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/phase2" 2>&1
sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/phase2" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/phase2"
grep -q "resumed" "$TMP/phase2"
check "phase 2: resumed notice printed" $?
grep -q "resume-replay-probe-7z9" "$TMP/phase2"
check "phase 2: the old prompt is shown" $?
grep -q "PLAIN-OK" "$TMP/phase2"
check "phase 2: the old answer is shown" $?
grep -q "end of resumed transcript" "$TMP/phase2"
check "phase 2: replay is delimited from new conversation" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT RESUME REPLAY E2E: ALL PASSED"
  exit 0
fi
echo "CHAT RESUME REPLAY E2E: $FAILURES FAILURE(S) — phase outputs follow"
echo "===== phase 1 ====="; cat "$TMP/phase1"
echo "===== phase 2 ====="; cat "$TMP/phase2"
exit 1
