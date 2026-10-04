#!/usr/bin/env bash
# tests/chat_resume_replay_e2e.sh — resuming distills through the model.
#
# /resume must NOT dump the old transcript: with 2+ turns and a usable
# provider it asks the ACTIVE model for a brief continuity summary, which
# becomes the context (plus the last exchange verbatim). Only a single
# turn, or no provider to distill with, falls back to the direct load.
#
# Phase 1: two mock turns in a fixture project, then grab that session's
# short id from the sync line.
# Phase 2: a fresh chat runs /resume <short> and must show the summary
# (the mock's reply) and the LAST exchange — but NOT the first turn.
# Phase 3: a provider-less config resumes the same session and must fall
# back to the full transcript (both prompts, end marker, no summary).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18890 + RANDOM % 200))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

# Resuming needs the session store, which needs the QTSQ codec: without it
# persist fails ("transcript is memory-only") and no session ever owns
# turns, so phase 2 can never pass. CI builds QTSQ-free on purpose (the
# codec is a proprietary local checkout). Exit 77 is the POSIX skip
# convention run_js_tests.sh honours. Same probe as chat_prompt_dedupe —
# one codec gates both.
if ! "$SOFUU" eval 'console.log("__HAS_MEMORY__" + (typeof sofuu.memory === "object" && typeof sofuu.memory.open === "function"))' \
      2>/dev/null | grep -q '__HAS_MEMORY__true'; then
  echo "SKIP chat_resume_replay_e2e — this build has no QTSQ codec, so sessions never persist turns"
  exit 77
fi

TMP="$(mktemp -d /tmp/sofuu_resume_replay_XXXXXX)"
HOME_DIR="$TMP/home"
HOME_DEAD="$TMP/home_dead"
PROJ="$TMP/proj"
mkdir -p "$HOME_DIR/.sofuu" "$HOME_DEAD/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; wait $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF
# Same provider shape, dead endpoint: the distill call fails fast and the
# resume must fall back to the direct load (no wizard — a configured
# provider skips setup, so stdin reaches the command).
cat > "$HOME_DEAD/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "http://127.0.0.1:9/v1/chat/completions",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF
# Same provider shape, dead endpoint: the distill call fails fast and the
# resume must fall back to the direct load (no wizard — a configured
# provider skips setup, so stdin reaches the command).

( cd "$ROOT" && exec env HOME="$HOME_DIR" "$SOFUU" run tests/mock_llm_server.js "$PORT" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-LLM-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$TMP/mock.log" || { echo "mock LLM failed to start"; cat "$TMP/mock.log"; exit 1; }

strip() { # strip <file>: remove ANSI escapes in place (both sed flavors)
  sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$1" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$1"
}

# ── phase 1: two real turns, so the session owns a distillable past ──
printf 'resume-probe-first-7z9\nresume-probe-second-7z9\n/exit\n' |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/phase1" 2>&1
strip "$TMP/phase1"
[ "$(grep -c "PLAIN-OK" "$TMP/phase1")" -ge 2 ]
check "phase 1: two mock turns completed (session owns a past)" $?
SHORT="$(grep -o 'session s-[0-9a-f]* registered' "$TMP/phase1" | head -1 | awk '{print $2}')"
[ -n "$SHORT" ]
check "phase 1: short session id captured ($SHORT)" $?

# ── phase 2: fresh chat resumes it → summary, not a dump ──────────
printf '/resume %s\n/exit\n' "$SHORT" |
  HOME="$HOME_DIR" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/phase2" 2>&1
strip "$TMP/phase2"
grep -q "resumed" "$TMP/phase2"
check "phase 2: resumed notice printed" $?
grep -q "summary" "$TMP/phase2"
check "phase 2: notice says the past was distilled, not dumped" $?
grep -q "PLAIN-OK" "$TMP/phase2"
check "phase 2: the model's summary is shown" $?
! grep -q "resume-probe-first-7z9" "$TMP/phase2"
check "phase 2: the first turn is NOT dumped" $?
grep -q "resume-probe-second-7z9" "$TMP/phase2"
check "phase 2: the last exchange stays verbatim" $?
grep -q "full transcript: /context" "$TMP/phase2"
check "phase 2: points at the full transcript" $?

# ── phase 3: dead endpoint → direct-load fallback ──────────────────
printf '/resume %s\n/exit\n' "$SHORT" |
  HOME="$HOME_DEAD" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/phase3" 2>&1
strip "$TMP/phase3"
grep -q "resumed" "$TMP/phase3"
check "phase 3: resumed notice printed with a dead endpoint" $?
grep -q "summary unavailable" "$TMP/phase3"
check "phase 3: fallback says why it is showing the transcript" $?
grep -q "resume-probe-first-7z9" "$TMP/phase3"
check "phase 3: fallback shows the first turn" $?
grep -q "resume-probe-second-7z9" "$TMP/phase3"
check "phase 3: fallback shows the last turn" $?
grep -q "end of resumed transcript" "$TMP/phase3"
check "phase 3: fallback replay is delimited" $?
! grep -q "→ summary" "$TMP/phase3"
check "phase 3: no summary claimed without a model" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT RESUME REPLAY E2E: ALL PASSED"
  exit 0
fi
echo "CHAT RESUME REPLAY E2E: $FAILURES FAILURE(S) — phase outputs follow"
for p in phase1 phase2 phase3; do echo "===== $p ====="; cat "$TMP/$p"; done
exit 1
