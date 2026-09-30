#!/usr/bin/env bash
# tests/chat_mention_e2e.sh — E2E for @agent mentions in interactive chat.
#
# Spins an isolated HOME + a scripted mock LLM endpoint
# (tests/mock_llm_server.js), defines one agent (helper), then pipes
# prompts through `sofuu` chat:
#
#   1. "@helper report status"      → agent runs → HELPER-RAN-OK + via line
#   2. "@agent:helper report status"→ explicit form → HELPER-RAN-OK
#   3. "@nosuchname hello"          → falls through to @file path → the mock
#                                     sees the missing-file marker
#
# No network, no API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_mention_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18699 + RANDOM % 90))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

# ── isolated environment ─────────────────────────────────────────
TMP="$(mktemp -d /tmp/sofuu_chat_mention_XXXXXX)"
HOME_DIR="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_DIR/.sofuu/agents" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_DIR/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false }
EOF

cat > "$HOME_DIR/.sofuu/agents/helper.js" <<EOF
sofuu.agent.define({
  name: "helper",
  system: "You are AGENT=helper, a focused test agent.",
  memory: "off",
  provider: "custom", model: "mock", base_url: "$BASE", api_key: "x",
});
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
  ( cd "$PROJ" && HOME="$HOME_DIR" perl -e 'alarm 60; exec @ARGV' \
      "$SOFUU" chat >"$1" 2>&1 )
}

# ── 1. plain form: @helper <task> ────────────────────────────────
printf '@helper report status\n/exit\n' | chat_run "$TMP/out1"
grep -q "HELPER-RAN-OK" "$TMP/out1"; check "@helper runs the named agent" $?
grep -q "via helper" "$TMP/out1";    check "@helper prints the via-summary line" $?

# ── 2. explicit form: @agent:helper <task> ───────────────────────
printf '@agent:helper report status\n/exit\n' | chat_run "$TMP/out2"
grep -q "HELPER-RAN-OK" "$TMP/out2"; check "@agent:helper explicit form runs the agent" $?

# ── 3. unknown name falls through to the @file path ──────────────
printf '@nosuchname hello\n/exit\n' | chat_run "$TMP/out3"
grep -q "SAW-MISSING-MARKER" "$TMP/out3"; check "@unknown falls through to @file (missing) path" $?

# ── 4. real @file attachment content reaches the model ───────────
printf 'canary line FILE-CONTENT-CANARY-8321\n' > "$PROJ/canary.txt"
printf 'summarize @canary.txt\n/exit\n' | chat_run "$TMP/out4"
grep -q "SAW-FILE-CONTENT" "$TMP/out4"; check "@file attaches real file content (await fix)" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT MENTION E2E: ALL PASSED"
  exit 0
fi
echo "CHAT MENTION E2E: $FAILURES FAILURE(S)"
exit 1
