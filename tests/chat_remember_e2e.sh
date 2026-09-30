#!/usr/bin/env bash
# tests/chat_remember_e2e.sh — E2E for the /remember brain-path fix
# (2026-09-09): /remember used to open the driver's OWN brain handle at
# ~/.sofuu_brain.qtsq while every turn's recall/auto-store ran on the
# agent runtime's project-local <cwd>/.sofuu/brain/brain.qtsq — pins went
# to a file recall never read, and two handles on one file would clobber
# on flush (a QTSQ flush rewrites the whole file; last flush wins).
#
# The fix routes /remember, /share, /import and ghost completion through
# sofuu.agent.brainFor(def) — the SAME BRAINS-cached handle the turn loop
# uses — and adds a config.json "brain_path" override honored by both
# sides.
#
# Proven here on a real pty against the scripted mock LLM:
#   1. REMEMBER-STORE-LOCAL  — /remember lands in <proj>/.sofuu/brain/
#      (brain.qtsq exists) and the HOME file ~/.sofuu_brain.qtsq is NEVER
#      created;
#   2. REMEMBER-CROSS-SESSION — a second chat session in the SAME folder
#      gets the canary INJECTED INTO THE MODEL REQUEST (mock asserts the
#      RECALL-HIT reply — proof lives in the request log, not the TUI,
#      which would echo the question back);
#   3. REMEMBER-BRAIN-PATH-OVERRIDE — config brain_path points both
#      sides at a custom file: the pin lands there instead.
#
# (The fused SEM2 sibling brain-v2.qtsq is deliberately NOT asserted on a
# bare /remember: the v2 channel flushes on turn stores, not on driver
# pins — verified 2026-09-09. Fusion persistence is covered by the brain
# battery in make test.)
#
# No network, no real API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_remember_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18901 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() {
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_remember_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
PROJ2="$TMP/proj2"
MOCK_LOG="$TMP/mock.log"
mkdir -p "$HOME_P/.sofuu" "$PROJ" "$PROJ2"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

mkconfig() { # mkconfig <extra-json-fields>
  cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock", "base_url": "$BASE",
  "api_key": "x", "brain": true, "ghost": false, "sync": false, $1 }
EOF
}

# ── mock LLM (plain SSE replies; logs every request) ─────────────
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run tests/mock_llm_server.js "$PORT" ) > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-LLM-READY" "$MOCK_LOG" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$MOCK_LOG" || { echo "mock LLM failed to start"; cat "$MOCK_LOG"; exit 1; }

CANARY="REMEMBER-CANARY-QZ77 pin the deploy marker"
CUSTOM="$TMP/custom/brain.qtsq"

# ── drive one chat session on a pty ──────────────────────────────
# drive_chat <dir> <extra-cfg-json> <cmd> <wait-secs>. Prints the full
# TUI output to stdout.
drive_chat() {
  CHAT_DIR="$1" EXTRA_CFG="$2" CMD="$3" WAIT_SECS="$4" \
  HOME_P="$HOME_P" SOFUU="$SOFUU" \
  python3 - <<'PYEOF'
import os, pty, sys, time, select

home = os.environ["HOME_P"]
sofuu = os.environ["SOFUU"]
proj = os.environ["CHAT_DIR"]
cmd = os.environ["CMD"]
wait = float(os.environ.get("WAIT_SECS", "6"))

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = home
    os.environ["TERM"] = "xterm-256color"
    os.chdir(proj)
    os.execv(sofuu, [sofuu, "chat"])

out = b""
start = time.time()
sent = False
while time.time() - start < wait:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    if not sent and (time.time() - start) > 3:
        sent = True
        os.write(fd, cmd.encode() + b"\r")
    if time.time() - start > wait - 1.5:
        break
try:
    os.write(fd, b"\x03")
    time.sleep(0.3)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
except Exception:
    pass
sys.stdout.write(out.decode("utf-8", "replace"))
print("PTY-DONE")
PYEOF
}

# ── 1) /remember in PROJ: must land project-local ────────────────
mkconfig '"rlm": "off"'
echo "== session 1: /remember in $PROJ"
OUT1="$(drive_chat "$PROJ" '"rlm": "off"' "/remember $CANARY" 7)"
echo "$OUT1" | sed 's/^/  /' | head -30

grep -q "⏺ remembered" <<<"$OUT1"
check "/remember acked by TUI" $?

[ -f "$PROJ/.sofuu/brain/brain.qtsq" ]
check "project-local brain.qtsq created" $?

[ ! -e "$HOME_P/.sofuu_brain.qtsq" ]
check "HOME ~/.sofuu_brain.qtsq NOT created" $?

# ── 2) cross-session recall in the SAME folder ───────────────────
# The question text is deliberately canary-free (no QZ77 substring) so
# the ONLY way the mock can answer RECALL-HIT-QZ77 is a real recall hit
# injected into the request as shared-brain context — a retained-history
# echo can't fake it either (fresh process, first turn of a new session).
echo "== session 2: cross-session recall in $PROJ"
OUT2="$(drive_chat "$PROJ" '"rlm": "off"' "What is the pinned deploy marker? (needle: ZEE-77)" 10)"
echo "$OUT2" | sed 's/^/  /' | tail -20

grep -aq "RECALL-HIT-QZ77" <<<"$OUT2"
check "session 2 model reply = RECALL-HIT-QZ77 (recall injected)" $?

# Request-level ground truth: the mock ONLY replies RECALL-HIT-QZ77 when
# the canary text is present in the request payload (mock_llm_server.js
# matches REMEMBER-CANARY across all messages) — a TUI echo or retained
# history can't produce this line.
grep -q "MOCK-LLM: reply=RECALL-HIT-QZ77" "$MOCK_LOG"
check "mock log reply=RECALL-HIT-QZ77 (canary was in the request)" $?

# ── 3) brain_path override: pin lands in the custom file ─────────
echo "== session 3: brain_path override"
mkconfig '"rlm": "off", "brain_path": "'"$CUSTOM"'"'
OUT3="$(drive_chat "$PROJ2" '"rlm": "off", "brain_path": "'"$CUSTOM"'"' "/remember REMEMBER-OVERRIDE-PIN-LW55" 7)"
echo "$OUT3" | sed 's/^/  /' | tail -15

grep -q "⏺ remembered" <<<"$OUT3"
check "/remember acked under brain_path override" $?

[ -f "$CUSTOM" ]
check "custom brain_path file created" $?

[ ! -e "$PROJ2/.sofuu/brain/brain.qtsq" ]
check "project-local file NOT created when brain_path is set" $?

# ── summary ──────────────────────────────────────────────────────
if [ "$FAILURES" -eq 0 ]; then
  echo "chat_remember_e2e: ALL PASS"
  exit 0
else
  echo "chat_remember_e2e: $FAILURES FAILURES"
  exit 1
fi
