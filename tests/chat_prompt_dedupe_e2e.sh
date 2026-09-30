#!/usr/bin/env bash
# tests/chat_prompt_dedupe_e2e.sh — prompt-event dedupe E2E (session-2,
# AUDIT-2026-09-07). A logical turn can take several provider attempts
# (THINK: reasoning_effort rejected → one no-effort retry; CAPACITY:
# 401/429 → failover to the next provider), and turn() re-runs per
# attempt. The prompt event must be logged EXACTLY ONCE per logical turn —
# duplicates replay as extra user turns on resume — and a fresh submit
# must log again (no over-suppression).
#
# Four legs, all against tests/mock_prompt_dedupe_server.js (keys on the
# wire body: any request carrying "reasoning_effort" → 400 THINK-class;
# model mock-p1 → 401 CAPACITY-class; else SSE PLAIN-OK):
#   think    desktop JS driver — config model "o3" (caps ladder) + effort
#            high; asserts resume().turns === 1
#   think    TUI piped chat — two turns; `sofuu session show` must hold
#            exactly ONE prompt event per turn
#   capacity desktop JS driver — mock-p1 401s → failover to p2/mock-p2;
#            two submits assert turns 1 → 2 (no over-suppression)
#   capacity TUI piped chat — same failover; session show counts
#
# No network, no API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_prompt_dedupe_e2e.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}
strip_ansi() {
  sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$1" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$1"
}

TMP="$(mktemp -d /tmp/sofuu_prompt_dedupe_XXXXXX)"
MOCK_A1=""; MOCK_A2=""; MOCK_B1=""; MOCK_B2=""
trap 'for p in "$MOCK_A1" "$MOCK_A2" "$MOCK_B1" "$MOCK_B2"; do kill "$p" 2>/dev/null; done; rm -rf "$TMP"' EXIT

wait_ready() { # wait_ready <logfile> <marker>
  for _ in $(seq 1 50); do
    grep -q "$2" "$1" 2>/dev/null && return 0
    sleep 0.1
  done
  return 1
}

# ── Leg A1: THINK retry, desktop JS driver ─────────────────────────
PA1=$((18930 + RANDOM % 30))
HA1="$TMP/home_a1"; PROJA1="$TMP/proj_a1"
mkdir -p "$HA1/.sofuu" "$PROJA1"
cat > "$HA1/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "o3", "base_url": "http://127.0.0.1:$PA1/v1/chat/completions",
  "api_key": "x", "effort": "high", "ml": true, "brain": true,
  "providers": [ { "name": "custom", "model": "o3",
                   "endpoint": "http://127.0.0.1:$PA1/v1/chat/completions", "api_key": "x" } ],
  "active": "custom" }
EOF
"$SOFUU" run "$ROOT/tests/mock_prompt_dedupe_server.js" "$PA1" > "$TMP/mock_a1.log" 2>&1 &
MOCK_A1=$!
wait_ready "$TMP/mock_a1.log" "MOCK-DEDUPE-READY"
DEDUPE_PROJ="$PROJA1" DEDUPE_MODE=think HOME="$HA1" \
  "$SOFUU" run "$ROOT/tests/chat_prompt_dedupe_driver.js" "$PA1" > "$TMP/drv_a1.log" 2>&1
check "think desktop: driver exit 0" $?
grep -q "think: resume turns === 1" "$TMP/drv_a1.log"
check "think desktop: exactly one prompt event for the retrying turn" $?
[ "$(grep -c "model=o3 think=yes" "$TMP/mock_a1.log")" -eq 1 ]
check "think desktop: exactly one effortful wire attempt was rejected" $?
[ "$(grep -c "model=o3 think=no" "$TMP/mock_a1.log")" -eq 1 ]
check "think desktop: retry went out WITHOUT effort" $?

# ── Leg A2: THINK retry, TUI piped chat ────────────────────────────
PA2=$((18960 + RANDOM % 30))
HA2="$TMP/home_a2"; PROJA2="$TMP/proj_a2"
mkdir -p "$HA2/.sofuu" "$PROJA2"
cat > "$HA2/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "o3", "base_url": "http://127.0.0.1:$PA2/v1/chat/completions",
  "api_key": "x", "effort": "high", "brain": false, "ghost": false }
EOF
"$SOFUU" run "$ROOT/tests/mock_prompt_dedupe_server.js" "$PA2" > "$TMP/mock_a2.log" 2>&1 &
MOCK_A2=$!
wait_ready "$TMP/mock_a2.log" "MOCK-DEDUPE-READY"
printf 'DEDUPE-TURN-ONE please answer plainly\nDEDUPE-TURN-TWO and again\n/exit\n' |
  HOME="$HA2" SOFUU_PROJECT="$PROJA2" SOFUU_STREAM_RETRY_DELAYS="300,300" SOFUU_STALL_TIMEOUT_SECS=3 \
  "$SOFUU" chat > "$TMP/tui_a2.out" 2>&1
strip_ansi "$TMP/tui_a2.out"
grep -q "PLAIN-OK" "$TMP/tui_a2.out"
check "think TUI: both turns answered (PLAIN-OK visible)" $?
SID_A2=$(SOFUU_PROJECT="$PROJA2" "$SOFUU" session list 2>/dev/null | awk '/^  s-/{print $1; exit}')
[ -n "$SID_A2" ]; check "think TUI: session registered ($SID_A2)" $?
SOFUU_PROJECT="$PROJA2" "$SOFUU" session show "$SID_A2" > "$TMP/show_a2.out" 2>&1
strip_ansi "$TMP/show_a2.out"
[ "$(grep -c "prompt: DEDUPE-TURN-ONE" "$TMP/show_a2.out")" -eq 1 ]
check "think TUI: turn 1 logged exactly one prompt event (not one per retry)" $?
[ "$(grep -c "prompt: DEDUPE-TURN-TWO" "$TMP/show_a2.out")" -eq 1 ]
check "think TUI: turn 2 logged its own prompt event" $?
[ "$(grep -c "model=o3 think=yes" "$TMP/mock_a2.log")" -eq 1 ]
check "think TUI: exactly one effortful wire attempt was rejected" $?
[ "$(grep -c "model=o3 think=no" "$TMP/mock_a2.log")" -ge 2 ]
check "think TUI: retry + second turn went out WITHOUT effort" $?

# ── Leg B1: CAPACITY failover, desktop JS driver ───────────────────
PB1=$((19000 + RANDOM % 30))
HB1="$TMP/home_b1"; PROJB1="$TMP/proj_b1"
mkdir -p "$HB1/.sofuu" "$PROJB1"
cat > "$HB1/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-p1", "base_url": "http://127.0.0.1:$PB1/v1/chat/completions",
  "api_key": "x", "ml": true, "brain": true,
  "providers": [ { "name": "p1", "model": "mock-p1",
                   "endpoint": "http://127.0.0.1:$PB1/v1/chat/completions", "api_key": "x" },
                 { "name": "p2", "model": "mock-p2",
                   "endpoint": "http://127.0.0.1:$PB1/v1/chat/completions", "api_key": "x" } ],
  "active": "p1" }
EOF
"$SOFUU" run "$ROOT/tests/mock_prompt_dedupe_server.js" "$PB1" > "$TMP/mock_b1.log" 2>&1 &
MOCK_B1=$!
wait_ready "$TMP/mock_b1.log" "MOCK-DEDUPE-READY"
DEDUPE_PROJ="$PROJB1" DEDUPE_MODE=capacity HOME="$HB1" \
  "$SOFUU" run "$ROOT/tests/chat_prompt_dedupe_driver.js" "$PB1" > "$TMP/drv_b1.log" 2>&1
check "capacity desktop: driver exit 0" $?
grep -q "resume turns === 1 after the failover turn" "$TMP/drv_b1.log"
check "capacity desktop: one prompt event for the failover turn" $?
grep -q "resume turns === 2 after the second submit" "$TMP/drv_b1.log"
check "capacity desktop: second submit logged again (no over-suppression)" $?
[ "$(grep -c "model=mock-p1" "$TMP/mock_b1.log")" -eq 2 ]
check "capacity desktop: each submit hit the failing p1 once" $?
[ "$(grep -c "model=mock-p2" "$TMP/mock_b1.log")" -eq 2 ]
check "capacity desktop: each submit failed over to p2" $?
[ "$(grep -c "think=yes" "$TMP/mock_b1.log")" -eq 0 ]
check "capacity desktop: no reasoning_effort ever sent (unknown models)" $?

# ── Leg B2: CAPACITY failover, TUI piped chat ──────────────────────
PB2=$((19030 + RANDOM % 30))
HB2="$TMP/home_b2"; PROJB2="$TMP/proj_b2"
mkdir -p "$HB2/.sofuu" "$PROJB2"
cat > "$HB2/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-p1", "base_url": "http://127.0.0.1:$PB2/v1/chat/completions",
  "api_key": "x", "brain": false, "ghost": false,
  "providers": [ { "name": "p1", "model": "mock-p1",
                   "endpoint": "http://127.0.0.1:$PB2/v1/chat/completions", "api_key": "x" },
                 { "name": "p2", "model": "mock-p2",
                   "endpoint": "http://127.0.0.1:$PB2/v1/chat/completions", "api_key": "x" } ],
  "active": "p1" }
EOF
"$SOFUU" run "$ROOT/tests/mock_prompt_dedupe_server.js" "$PB2" > "$TMP/mock_b2.log" 2>&1 &
MOCK_B2=$!
wait_ready "$TMP/mock_b2.log" "MOCK-DEDUPE-READY"
printf 'DEDUPE-TURN-ONE first turn\nDEDUPE-TURN-TWO second turn\n/exit\n' |
  HOME="$HB2" SOFUU_PROJECT="$PROJB2" SOFUU_STREAM_RETRY_DELAYS="300,300" SOFUU_STALL_TIMEOUT_SECS=3 \
  "$SOFUU" chat > "$TMP/tui_b2.out" 2>&1
strip_ansi "$TMP/tui_b2.out"
grep -q "PLAIN-OK" "$TMP/tui_b2.out"
check "capacity TUI: turns answered via failover (PLAIN-OK visible)" $?
SID_B2=$(SOFUU_PROJECT="$PROJB2" "$SOFUU" session list 2>/dev/null | awk '/^  s-/{print $1; exit}')
[ -n "$SID_B2" ]; check "capacity TUI: session registered ($SID_B2)" $?
SOFUU_PROJECT="$PROJB2" "$SOFUU" session show "$SID_B2" > "$TMP/show_b2.out" 2>&1
strip_ansi "$TMP/show_b2.out"
[ "$(grep -c "prompt: DEDUPE-TURN-ONE" "$TMP/show_b2.out")" -eq 1 ]
check "capacity TUI: turn 1 logged exactly one prompt event (not one per provider)" $?
[ "$(grep -c "prompt: DEDUPE-TURN-TWO" "$TMP/show_b2.out")" -eq 1 ]
check "capacity TUI: turn 2 logged its own prompt event" $?
[ "$(grep -c "model=mock-p1" "$TMP/mock_b2.log")" -eq 2 ]
check "capacity TUI: each turn hit the failing p1 once" $?
[ "$(grep -c "model=mock-p2" "$TMP/mock_b2.log")" -eq 2 ]
check "capacity TUI: each turn failed over to p2" $?

echo ""
echo "===== driver logs (desktop legs) ====="
sed 's/^/  a1: /' "$TMP/drv_a1.log" | tail -8
sed 's/^/  b1: /' "$TMP/drv_b1.log" | tail -10
echo "===== session show (TUI legs) ====="
sed 's/^/  a2: /' "$TMP/show_a2.out"
sed 's/^/  b2: /' "$TMP/show_b2.out"
echo "===== mock logs (wire attempts) ====="
sed 's/^/  a1: /' "$TMP/mock_a1.log"
sed 's/^/  a2: /' "$TMP/mock_a2.log"
sed 's/^/  b1: /' "$TMP/mock_b1.log"
sed 's/^/  b2: /' "$TMP/mock_b2.log"

if [ "$FAILURES" -eq 0 ]; then
  echo ""
  echo "CHAT PROMPT-DEDUPE E2E: ALL PASSED"
  exit 0
fi
echo ""
echo "CHAT PROMPT-DEDUPE E2E: $FAILURES FAILURE(S) — logs above"
exit 1
