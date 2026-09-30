#!/usr/bin/env bash
# tests/chat_no_response_e2e.sh — what does the chat show when the provider
# fails? Runs the REAL chat binary against tests/mock_fail_server.js in six
# failure modes and HARD-ASSERTS what the user would see (P1-17,
# AUDIT-2026-09-01: the script used to print-and-exit-0 — unfalsifiable).
# SOFUU_STALL_TIMEOUT_SECS=3 shrinks the 5-minute unresponsive-model patience
# window so the MODESILENT turn aborts in ~4s instead of ~5min.
# SOFUU_CHAT_MAX_STEPS=2 shrinks the chat step budget so the MODELOOP turn
# breaches after two tool rounds — the chat must print the salvage summary
# plus a "⏹ stopped: step budget (2 rounds) reached" line, never a bare
# "(no response)".
# MODECLOSE must be the LAST mode turn: it kills the mock process (zero
# response bytes → curl transport error, e.g. "Empty reply from server").
# The retry classifier must treat that as transient: the chat retries
# against the now-dead port (SOFUU_STREAM_RETRY_DELAYS pins a short
# 2x0.3s ladder — the shipped default is a patient 6-step backoff meant
# for riding out real provider outages, far too slow for this suite) and
# fails loudly — the final "Couldn't connect" proves attempt 1 was
# classified transient and retried instead of surfacing immediately.
# SOFUU_STREAM_RETRY_MAX_CONT pins continuation attempts (default 3).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18790 + RANDOM % 200))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_noresp_XXXXXX)"
HOME_DIR="$TMP/home"
mkdir -p "$HOME_DIR/.sofuu" "$TMP/proj"
trap 'kill $MOCK_PID 2>/dev/null; wait $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

"$SOFUU" run "$ROOT/tests/mock_fail_server.js" "$PORT" >"$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 50); do grep -q "MOCK-FAIL-READY" "$TMP/mock.log" 2>/dev/null && break; sleep 0.1; done

printf 'MODE429 hello\nMODEERRFRAME hello\nMODEEMPTY hello\nMODESILENT hello\nMODELOOP hello\nMODECLOSE hello\n/exit\n' |
  HOME="$HOME_DIR" SOFUU_PROJECT="$TMP/proj" SOFUU_STALL_TIMEOUT_SECS=3 SOFUU_CHAT_MAX_STEPS=2 \
  SOFUU_STREAM_RETRY_DELAYS="300,300" \
  "$SOFUU" chat >"$TMP/out" 2>&1
sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/out" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/out"

# ── hard assertions on all six failure modes ───────────────────────
# MODE429 and MODEERRFRAME both end in the same loud "✗ HTTP 429" line
# (the errframe arrives as HTTP 200 + an in-stream error object carrying
# code 429 — one final ✗ per turn after the transient retries), so the
# per-mode attribution comes from the mock log + the line count.
N429="$(grep -c "✗ HTTP 429" "$TMP/out" || true)"
[ "$N429" -eq 2 ]; check "MODE429 + MODEERRFRAME: both turns end in the loud HTTP 429 error" $?
grep -q "mode=429" "$TMP/mock.log"; check "MODE429: mock served the 429 status path" $?
grep -q "mode=errframe" "$TMP/mock.log"; check "MODEERRFRAME: mock served the in-stream error frame path" $?
grep -q "provider returned an empty stream" "$TMP/out"
check "MODEEMPTY: empty stream surfaces with its cause, never a bare (no response)" $?
grep -q "provider sent no data for 3s" "$TMP/out"
check "MODESILENT: stall watchdog fires (unresponsive model)" $?
grep -q "SALVAGED-SUMMARY" "$TMP/out"
check "MODELOOP: salvage summary present at step-budget breach" $?
grep -q "stopped: step budget (2 rounds) reached" "$TMP/out"
check "MODELOOP: honest stop line (budget reached)" $?
grep -q "Couldn't connect to server" "$TMP/out"
check "MODECLOSE: dead-port transport error after transient retries" $?
! grep -q "(no response)" "$TMP/out"
check "no turn anywhere fell back to the bare (no response) placeholder" $?
grep -q "MOCK-FAIL-READY" "$TMP/mock.log"
check "mock server was up for the whole run" $?

echo ""
echo "===== chat output (what the user sees) ====="
cat "$TMP/out"
echo "===== mock server log ====="
cat "$TMP/mock.log"

if [ "$FAILURES" -eq 0 ]; then
  echo ""
  echo "CHAT NO-RESPONSE E2E: ALL PASSED"
  exit 0
fi
echo ""
echo "CHAT NO-RESPONSE E2E: $FAILURES FAILURE(S) — chat output above"
exit 1
