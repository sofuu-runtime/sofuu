#!/usr/bin/env bash
# tests/chat_paste_e2e.sh — E2E for bracketed-paste input (DECSET 2004).
#
# The TUI readline used to treat every CR/LF byte inside a pasted block
# as Enter: a large multi-line paste auto-submitted line after line in
# batches and anything past the 8 KiB line cap was silently dropped.
# The fix enables bracketed paste (ESC[?2004h) on TTY open, buffers the
# paste body (ESC[200~ … ESC[201~) as ONE multi-line prompt whose embedded
# newlines never submit, and raises the line cap to 1 MiB.
#
# This suite drives the chat binary on a real pty (python pty.fork — the
# fix only exists on the raw-mode path) against the scripted mock LLM
# (tests/mock_llm_server.js) which logs every request:
#   1. PASTE-NO-AUTO-SUBMIT — a 300-line (~30 KB) wrapped paste makes ZERO
#      model requests before Enter;
#   2. PASTE-WHOLE-ONE-REQUEST — after Enter, exactly ONE request whose
#      user message holds the whole ~30 KB (first + last canary present);
#   3. TYPED-STILL-SUBMITS — a short typed line afterwards still submits
#      normally on Enter.
#
# No network, no real API keys. Exits non-zero on any failure.
# Run:  bash tests/chat_paste_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((18801 + RANDOM % 89))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

# ── isolated environment ─────────────────────────────────────────
TMP="$(mktemp -d /tmp/sofuu_chat_paste_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
MOCK_LOG="$TMP/mock.log"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "sync": false }
EOF

# ── mock LLM (logs every request with sizes + heads) ─────────────
# exec: the subshell replaces itself with sofuu, so $! is the SERVER pid —
# a plain `( ... ) &` made MOCK_PID the subshell, the trap killed the shell,
# and the LISTENing sofuu leaked, poisoning the random port pool for every
# later script.
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run tests/mock_llm_server.js "$PORT" ) > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-LLM-READY" "$MOCK_LOG" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-LLM-READY" "$MOCK_LOG" || { echo "mock LLM failed to start"; cat "$MOCK_LOG"; exit 1; }

# ── the paste payload: 300 lines × ~100 bytes ≈ 30 KB ───────────
PASTE_FILE="$TMP/paste.txt"
: > "$PASTE_FILE"
for i in $(seq -w 1 300); do
  printf 'PASTED-LINE-%s- %s\n' "$i" "$(printf 'x%.0s' $(seq 1 80))" >> "$PASTE_FILE"
done
PASTE_BYTES=$(wc -c < "$PASTE_FILE" | tr -d ' ')

# ── drive the chat on a pty (python inline; no scratch files) ────
export PASTE_FILE HOME_P SOFUU MOCK_LOG
python3 - <<'PYEOF'
import os, pty, sys, time, select

home = os.environ["HOME_P"]
paste_file = os.environ["PASTE_FILE"]
sofuu = os.environ["SOFUU"]
proj = os.path.dirname(home) + "/proj"

payload = open(paste_file, "rb").read()
# CRLF line endings + bracketed-paste markers, exactly as macOS Terminal
# delivers a paste after DECSET 2004.
wrapped = b"\x1b[200~" + payload.replace(b"\n", b"\r\n") + b"\x1b[201~"

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = home
    os.environ["TERM"] = "xterm-256color"
    os.chdir(proj)
    os.execv(sofuu, [sofuu, "chat"])

out = b""
start = time.time()
pasted = entered = typed = entered2 = False
req0 = 0
while time.time() - start < 40:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    # 1) big paste at t=3 — must NOT submit by itself
    if not pasted and time.time() - start > 3:
        pasted = True
        os.write(fd, wrapped)
    # 2) wait 6s: a broken build would have submitted many fragments
    if pasted and not entered and time.time() - start > 9:
        entered = True
        os.write(fd, b"\r")
    # 3) typed control line after the answer
    if entered and not typed and time.time() - start > 13:
        typed = True
        os.write(fd, b"typed control line")
    if typed and not entered2 and time.time() - start > 15:
        entered2 = True
        os.write(fd, b"\r")
    if time.time() - start > 22:
        break
try:
    os.write(fd, b"\x03")
    time.sleep(0.3)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
except Exception:
    pass

# Signal pass/fail through stdout markers consumed by the shell layer.
text = out.decode("utf-8", "replace")
print("PTY-DONE")
PYEOF

# ── assertions from the mock's request log ──────────────────────
# The mock log line for each request carries sizes=… and heads=…; the
# paste canaries must appear in ONE request's payload (heads), and the
# request count must be exactly 2 (paste turn + typed turn).
REQ_COUNT=$(grep -c "MOCK-LLM:" "$MOCK_LOG")
# The LAST user:N in sizes= is the prompt itself (earlier ones are the
# ephemeral date/context message).
FIRST_USER=$(grep "MOCK-LLM:" "$MOCK_LOG" | head -1 | grep -o "user:[0-9]*" | tail -1 | cut -d: -f2)
HAS_FIRST=$(grep -q "PASTED-LINE-001" "$MOCK_LOG" && echo 1 || echo 0)
HAS_TYPED=$(grep -q "typed control line" "$MOCK_LOG" && echo 1 || echo 0)

check "exactly 2 requests (whole paste + typed control), got $REQ_COUNT" \
  $([ "$REQ_COUNT" -eq 2 ] && echo 0 || echo 1)
# The mock's heads= preview only shows the first 100 chars of each message,
# so the tail-line canary can only be proven via the full payload size:
# the request must carry the entire paste (>= its on-disk byte count) plus
# the date-message overhead — truncation at the old 8 KiB cap would fail.
check "first request carries the whole ~${PASTE_BYTES}B paste (user msg ${FIRST_USER}B, want >=${PASTE_BYTES})" \
  $([ -n "$FIRST_USER" ] && [ "$FIRST_USER" -ge "$PASTE_BYTES" ] && echo 0 || echo 1)
check "paste head canary (line 001) reached the model" $([ "$HAS_FIRST" -eq 1 ] && echo 0 || echo 1)
check "typed control line still submits (no input freeze)" $([ "$HAS_TYPED" -eq 1 ] && echo 0 || echo 1)

if [ "$FAILURES" -eq 0 ]; then
  echo "chat_paste_e2e: ALL PASS (${PASTE_BYTES}B paste, $REQ_COUNT requests)"
  exit 0
else
  echo "chat_paste_e2e: $FAILURES FAILURES"
  echo "--- mock log (first 2KB) ---"; head -c 2048 "$MOCK_LOG"
  exit 1
fi
