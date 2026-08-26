#!/usr/bin/env bash
# tests/chat_compact_e2e.sh — E2E for context-window management:
# the P5 one-shot cliff (PLAN-MEMORY-TOKENS) AND the §12 ML compaction
# gate (PLAN-ML-GATES).
#
# Same scripted scenario twice (isolated HOME per run + scripted mock
# LLM from tests/mock_llm_server.js, small ctx_window so thresholds are
# reachable, 10 padded prompts ≈ 500 tk each):
#
#   Run A (SOFUU_NO_ML=1) — the cliff ALONE, as before:
#     1. history grows past 70% of the budget → "auto-compacted → summary"
#        fires and the window resets to ~0
#     2. the summarizer call reaches the mock (extra non-stream request)
#
#   Run B (ML on) — the §12 gate:
#     3. repeated prompt + identical PLAIN-OK reply = dup turns; the gate
#        frees them opportunistically ("ml-compaction freed" lines) while
#        usage is past 50% of the budget
#     4. the drain keeps usage below the 70% cliff → the cliff never fires
#     5. the free tier is LLM-free: exactly one mock request per prompt,
#        zero summarizer calls
#     6. every turn still gets an answer — the gate never blocks a turn
#
# No network, no API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_compact_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-18701}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

# ── isolated environment (separate HOME per run) ─────────────────
TMP="$(mktemp -d /tmp/sofuu_chat_compact_XXXXXX)"
HOME_A="$TMP/home_a"   # run A: cliff alone (SOFUU_NO_ML=1)
HOME_B="$TMP/home_b"   # run B: §12 ML gate on
PROJ="$TMP/proj"
mkdir -p "$HOME_A/.sofuu" "$HOME_B/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

# Small context window: budget = 4000*0.85 = 3400 tk; 70% = 2380 tk,
# 50% = 1700 tk. Each turn adds ~510 tk of history (~2000-char prompt +
# PLAIN-OK). Run A: turn ~5 crosses 70% → the cliff summarizes the whole
# context (window → ~0) and may compact again as history regrows. Run B:
# the gate starts freeing dup turns past 50% and keeps usage oscillating
# below the cliff, so the one-shot never fires.
for H in "$HOME_A" "$HOME_B"; do
cat > "$H/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "ctx_window": 4000 }
EOF
done

# ── mock LLM endpoint (stateless, serves both runs) ──────────────
"$SOFUU" run "$ROOT/tests/mock_llm_server.js" "$PORT" >"$TMP/mock.log" 2>&1 &
MOCK_PID=$!

for _ in $(seq 1 50); do
  grep -q "MOCK-LLM-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$TMP/mock.log" || { echo "FAIL mock server did not start"; exit 1; }

# Build 10 long prompts (~2000 chars each ≈ 500 tk). The prompts differ
# only in the leading counter, so every prompt is a near-dup of the ones
# still in the recent window and every answer is the identical PLAIN-OK —
# exactly the mechanical junk the §12 free tier targets.
PROMPTS=""
for i in $(seq 1 10); do
  PROMPTS="${PROMPTS}question number $i with padding $(printf 'x%.0s' $(seq 1 2000))\n"
done

# ── run A: cliff alone (ML disabled) ─────────────────────────────
printf "${PROMPTS}/exit\n" | ( cd "$PROJ" && HOME="$HOME_A" SOFUU_NO_ML=1 \
  perl -e 'alarm 90; exec @ARGV' "$SOFUU" chat >"$TMP/out_a" 2>&1 )
REQ_A=$(grep -c "MOCK-LLM" "$TMP/mock.log")

# A1. auto-compaction fired
grep -q "auto-compacted" "$TMP/out_a"; check "A1 P5 cliff fires past 70% budget (ML off)" $?

# A2. the summarizer reached the mock (one extra non-stream request)
[ "$REQ_A" -ge 11 ]; check "A2 summarizer call reached the mock ($REQ_A requests ≥ 11)" $?

# A3. at least one compaction and window reset
COMPACT_COUNT=$(grep -c "auto-compacted" "$TMP/out_a")
[ "$COMPACT_COUNT" -ge 1 ]; check "A3 cliff fired at least once (got $COMPACT_COUNT)" $?

# A4. the cliff line shows the footer meter DROPPING ("meter A→B", A > B) —
# the meter measures CURRENT context usage, so compaction must shrink it.
METER_DROPS=0
while IFS= read -r m; do
  B=${m#meter }; A=${B%→*}; C=${B#*→}
  if awk -v a="$A" -v b="$C" 'BEGIN{
       af = (a ~ /k$/) ? substr(a, 1, length(a) - 1) * 1000 : a;
       bf = (b ~ /k$/) ? substr(b, 1, length(b) - 1) * 1000 : b;
       exit (af > bf) ? 0 : 1 }'; then
    METER_DROPS=$((METER_DROPS+1))
  fi
done < <(grep -oE 'meter [0-9.]+k?→[0-9.]+k?' "$TMP/out_a")
[ "$METER_DROPS" -ge 1 ]; check "A4 cliff line shows the meter dropping ($METER_DROPS drops ≥ 1)" $?

# ── run B: §12 ML compaction gate ────────────────────────────────
printf "${PROMPTS}/exit\n" | ( cd "$PROJ" && HOME="$HOME_B" \
  perl -e 'alarm 90; exec @ARGV' "$SOFUU" chat >"$TMP/out_b" 2>&1 )
REQ_B=$(( $(grep -c "MOCK-LLM" "$TMP/mock.log") - REQ_A ))

# B1. the gate freed junk turns opportunistically
ML_LINES=$(grep -c "ml-compaction freed" "$TMP/out_b")
[ "$ML_LINES" -ge 3 ]; check "B1 ml-compaction gate fired ($ML_LINES passes ≥ 3)" $?

# B2. the drain kept usage below the cliff — it never fired
CLIFF_COUNT=$(grep -c "auto-compacted" "$TMP/out_b")
[ "$CLIFF_COUNT" -eq 0 ]; check "B2 cliff never needed (got $CLIFF_COUNT, want 0)" $?

# B3. free tier is LLM-free: one mock request per prompt, no summarizer
[ "$REQ_B" -eq 10 ]; check "B3 zero extra LLM calls ($REQ_B requests = 10 prompts)" $?

# B4. every turn still answered — the gate never blocks or drops a turn
ANSWERS=$(grep -c "PLAIN-OK" "$TMP/out_b")
[ "$ANSWERS" -ge 10 ]; check "B4 all 10 turns answered ($ANSWERS replies ≥ 10)" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT COMPACT E2E: ALL PASSED"
  exit 0
fi
echo "CHAT COMPACT E2E: $FAILURES FAILURE(S)"
exit 1
