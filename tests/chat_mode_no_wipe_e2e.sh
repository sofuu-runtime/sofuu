#!/usr/bin/env bash
# tests/chat_mode_no_wipe_e2e.sh — a settings change must not clear chat.
#
# Regression: every settings command (/mode, /theme, /model, …) ended in
# __chat_refresh(), which tui_reset()s the scrollback — switching modes
# mid-conversation erased the visible transcript and looked like a fresh
# session. Refresh is now gated to pre-first-turn (maybeRefreshPanel).
#
# Drives the REAL chat on a pty: one mock turn, then /mode edit. The
# welcome panel is logged once at boot; a destructive refresh would log
# it a second time. So: exactly one "Welcome to Sofuu!" plus the mode
# confirmation proves the transcript survived.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29800 + RANDOM % 500))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_nowipe_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# ── mock LLM: models endpoint + one text answer ─────────────────
cat > "$TMP/mock.js" <<'MOCKEOF'
const server = sofuu.http.createServer(function (req, res) {
  if (String(req.url || "").indexOf("/models") >= 0) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ object: "list", data: [{ id: "live-alpha", context_length: 1048576, max_completion_tokens: 65536 }] }));
    return;
  }
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  const C = (d, f) => ({ id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: d, finish_reason: f }] });
  const W = (o) => res.write("data: " + JSON.stringify(o) + "\n\n");
  W(C({ content: "TURN-DONE" }, null));
  W(C({}, "stop"));
  res.write("data: [DONE]\n\n"); res.end();
});
server.listen(__TX_PORT__, "127.0.0.1");
console.log("MOCK-NOWIPE-READY");
MOCKEOF
sed -i '' "s/__TX_PORT__/$PORT/" "$TMP/mock.js" 2>/dev/null || sed -i "s/__TX_PORT__/$PORT/" "$TMP/mock.js"
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run "$TMP/mock.js" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-NOWIPE-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-NOWIPE-READY" "$TMP/mock.log" || { echo "mock failed to start"; cat "$TMP/mock.log"; exit 1; }

# ── pty: one turn, then switch modes ─────────────────────────────
export HOME_P SOFUU PROJ
python3 - <<'PYEOF'
import os, pty, time, select

home = os.environ["HOME_P"]
sofuu = os.environ["SOFUU"]
proj = os.environ["PROJ"]

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = home
    os.environ["TERM"] = "xterm-256color"
    os.environ["SOFUU_PROJECT"] = proj
    os.chdir(proj)
    os.execv(sofuu, [sofuu, "chat"])

out = b""
start = time.time()
s1 = s2 = False
while time.time() - start < 25:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    if not s1 and time.time() - start > 6:
        s1 = True
        os.write(fd, b"do the thing\r")
    if s1 and not s2 and time.time() - start > 14:
        s2 = True
        os.write(fd, b"/mode edit\r")
    if time.time() - start > 19:
        break
try:
    os.write(fd, b"/exit\r")
    time.sleep(0.5)
    os.write(fd, b"\x03")
    time.sleep(0.3)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
except Exception:
    pass

text = out.decode("utf-8", "replace")
with open(os.path.dirname(home) + "/pty.txt", "w") as f:
    f.write(text)
print("PTY-DONE")
PYEOF

PTY_TXT="$TMP/pty.txt"
FRAME_TXT="$TMP/final_frame.txt"

# ── final-frame reconstruction ───────────────────────────────
# The stream repaints rows constantly, so byte counts prove nothing. What
# matters is the last frame the user sees: rebuild it last-write-wins
# (absolute row positioning + erase-line, which is all this TUI emits)
# and assert the turn's answer is still on screen after the mode switch.
# A destructive refresh would leave only the re-logged panel + the mode
# confirmation — the answer rows would be gone from the buffer and could
# never repaint.
python3 - "$PTY_TXT" "$FRAME_TXT" <<'PYEOF'
import re, sys
raw = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
rows = {}
r, c = 1, 0
cur = []
i, n = 0, len(raw)
while i < n:
    ch = raw[i]
    if ch == "\x1b" and i + 1 < n and raw[i + 1] == "[":
        m = re.match(r"\x1b\[(\d+);(\d+)H", raw[i:])
        if m:
            r, c = int(m.group(1)), int(m.group(2)) - 1
            cur = rows.setdefault(r, [])
            i += m.end()
            continue
        if raw.startswith("\x1b[K", i):
            cur = rows.setdefault(r, [])
            del cur[c:]
            i += 3
            continue
        m2 = re.match(r"\x1b\[[0-9;?]*[a-zA-Z]", raw[i:])
        if m2:
            i += m2.end()
            continue
        i += 2
        continue
    if ch == "\n":
        r += 1; c = 0; cur = rows.setdefault(r, [])
        i += 1
        continue
    if ch == "\r":
        c = 0
        i += 1
        continue
    if ch == "\x00" or ord(ch) < 32:
        i += 1
        continue
    cur = rows.setdefault(r, [])
    while len(cur) < c:
        cur.append(" ")
    if c < len(cur):
        cur[c] = ch
    else:
        cur.append(ch)
    c += 1
    i += 1
with open(sys.argv[2], "w") as f:
    for rn in sorted(rows):
        f.write("".join(rows[rn]).rstrip() + "\n")
PYEOF

# ── assertions ───────────────────────────────────────────────────
grep -q "TURN-DONE" "$PTY_TXT"
check "the mock turn completed" $?
grep -q "mode → edit" "$PTY_TXT"
check "the mode switch applied" $?
grep -q "TURN-DONE" "$FRAME_TXT"
check "answer still on screen after the switch (no wipe)" $?
grep -q "mode → edit" "$FRAME_TXT"
check "mode confirmation on screen" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT MODE NO-WIPE E2E: ALL PASSED"
  exit 0
fi
echo "CHAT MODE NO-WIPE E2E: $FAILURES FAILURE(S) — pty transcript follows"
cat "$PTY_TXT"
exit 1
