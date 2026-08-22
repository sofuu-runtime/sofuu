#!/usr/bin/env bash
# tests/chat_compact_e2e.sh — E2E for P5 auto-compaction (PLAN-MEMORY-TOKENS).
#
# Spins an isolated HOME + a scripted mock LLM endpoint
# (tests/mock_llm_server.js), configures a SMALL ctx_window so the 70%
# auto-compaction threshold is reachable within a few turns, then pipes
# enough prompts through `sofuu` chat to cross it:
#
#   1. history grows past 70% of the budget → auto-compaction fires
#      (the "⚙ auto-compacted → summary" line appears)
#   2. the summarizer call reaches the mock (one extra non-stream request)
#   3. the one-shot guard: the very next turn does NOT compact again
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

# ── isolated environment ─────────────────────────────────────────
TMP="$(mktemp -d /tmp/sofuu_chat_compact_XXXXXX)"
HOME_DIR="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_DIR/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

# Small context window: budget = 4000*0.85 = 3400 tk; 70% = 2380 tk, 50% = 1700 tk.
# Each turn adds ~500 tk of history (~2000 chars). Turn ~5 crosses 70% →
# compaction fires, leaving summary + last-4-turns ≈ 2000 tk, which is STILL
# above 50% — so the one-shot guard stays disarmed and the later crossings
# (turns ~7-9 also pass 70%) must NOT re-compact. This is the real guard
# scenario: exactly one "auto-compacted" line for the whole session.
cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "ctx_window": 4000 }
EOF

# ── mock LLM endpoint ────────────────────────────────────────────
"$SOFUU" run "$ROOT/tests/mock_llm_server.js" "$PORT" >"$TMP/mock.log" 2>&1 &
MOCK_PID=$!

for _ in $(seq 1 50); do
  grep -q "MOCK-LLM-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$TMP/mock.log" || { echo "FAIL mock server did not start"; exit 1; }

chat_run() { # chat_run <outfile> ...prompts on stdin
  ( cd "$PROJ" && HOME="$HOME_DIR" perl -e 'alarm 90; exec @ARGV' \
      "$SOFUU" chat >"$1" 2>&1 )
}

# Build 10 long prompts (~2000 chars each ≈ 500 tk) to push history past 70%
# around turn 5, leaving turns 6-10 above the threshold to prove the guard.
PROMPTS=""
for i in $(seq 1 10); do
  PROMPTS="${PROMPTS}question number $i with padding $(printf 'x%.0s' $(seq 1 2000))\n"
done
printf "${PROMPTS}/exit\n" | chat_run "$TMP/out1"

# 1. auto-compaction fired
grep -q "auto-compacted" "$TMP/out1"; check "P5 auto-compaction fires past 70% budget" $?

# 2. the summarizer reached the mock (a non-stream completion request).
#    The mock logs one line per request; the summarizer is the only
#    non-stream caller in this session.
SUMMARIZER_CALLS=$(grep -c "MOCK-LLM" "$TMP/mock.log" 2>/dev/null || echo 0)
[ "$SUMMARIZER_CALLS" -ge 11 ]; check "P5 summarizer call reached the mock ($SUMMARIZER_CALLS requests ≥ 11)" $?

# 3. one-shot guard: count auto-compact lines — must be exactly 1 even
#    though history stays hot for several more turns.
COMPACT_COUNT=$(grep -c "auto-compacted" "$TMP/out1" 2>/dev/null || echo 0)
[ "$COMPACT_COUNT" -eq 1 ]; check "P5 one-shot guard: exactly one auto-compaction (got $COMPACT_COUNT)" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT COMPACT E2E: ALL PASSED"
  exit 0
fi
echo "CHAT COMPACT E2E: $FAILURES FAILURE(S)"
exit 1
