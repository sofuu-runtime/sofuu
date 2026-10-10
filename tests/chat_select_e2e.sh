#!/usr/bin/env bash
# tests/chat_select_e2e.sh — mouse-drag selection wears the theme accent.
#
# Regression (2026-10-10, live session): a drag-select painted raw inverse
# video (`\x1b[7m`) — black boxes on light themes, white boxes on dark
# ones, blending with nothing. Selection now wears the theme accent
# background + near-black text (opencode-style accent bar), derived from
# the accent role so it follows every theme with no table entry.
#
# Drives the REAL chat on a pty with SGR mouse reports: drag panel rows,
# assert the accent bar (not inverse); switch theme, drag again, assert
# the bar follows the new accent.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29900 + RANDOM % 400))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_select_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# ── mock LLM: models endpoint only (no turns needed) ────────────
cat > "$TMP/mock.js" <<'MOCKEOF'
const server = sofuu.http.createServer(function (req, res) {
  if (String(req.url || "").indexOf("/models") >= 0) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ object: "list", data: [{ id: "live-alpha", context_length: 1048576, max_completion_tokens: 65536 }] }));
    return;
  }
  res.writeHead(200, { "Content-Type": "application/json" });
  res.send(JSON.stringify({ choices: [{ message: { role: "assistant", content: "UNUSED" } }] }));
});
server.listen(__TX_PORT__, "127.0.0.1");
console.log("MOCK-SELECT-READY");
MOCKEOF
sed -i '' "s/__TX_PORT__/$PORT/" "$TMP/mock.js" 2>/dev/null || sed -i "s/__TX_PORT__/$PORT/" "$TMP/mock.js"
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run "$TMP/mock.js" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-SELECT-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-SELECT-READY" "$TMP/mock.log" || { echo "mock failed to start"; cat "$TMP/mock.log"; exit 1; }

# ── pty: drag, switch theme, drag again ──────────────────────────
export HOME_P SOFUU PROJ
python3 - <<'PYEOF'
import os, pty, time, select

home = os.environ["HOME_P"]
sofuu = os.environ["SOFUU"]
proj = os.environ["PROJ"]

def mouse(ev, x, y):
    tail = "m" if ev == 3 else "M"
    return ("\x1b[<%d;%d;%d%s" % (ev, x, y, tail)).encode()

def drag(fd, r1, r2):
    os.write(fd, mouse(0, 10, r1))
    time.sleep(0.2)
    os.write(fd, mouse(32, 60, r2))
    time.sleep(0.2)
    os.write(fd, mouse(3, 60, r2))
    time.sleep(0.3)

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = home
    os.environ["TERM"] = "xterm-256color"
    os.environ["SOFUU_PROJECT"] = proj
    os.chdir(proj)
    os.execv(sofuu, [sofuu, "chat"])

out = b""
start = time.time()
s1 = s2 = s3 = s4 = False
while time.time() - start < 38:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    t = time.time() - start
    if not s1 and t > 7:
        s1 = True
        drag(fd, 5, 9)
    if s1 and not s2 and t > 13:
        s2 = True
        os.write(fd, b"/theme slate\r")
    if s2 and not s3 and t > 18:
        s3 = True
        drag(fd, 5, 9)
    if s3 and not s4 and t > 24:
        s4 = True
        os.write(fd, b"/exit\r")
    if t > 32:
        break
try:
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

# ── assertions ───────────────────────────────────────────────────
# default theme accent is 139: the drag must wear 38;5;16 on 48;5;139.
grep -qF "$(printf '\x1b[38;5;16;48;5;139m')" "$PTY_TXT"
check "drag selection wears the default accent bar" $?
# raw inverse must be gone from the session (it painted black boxes on
# light themes and white boxes on dark ones).
if grep -q '\[7m' "$PTY_TXT"; then
  echo "FAIL raw inverse still emitted (selection does not blend)"
  FAILURES=$((FAILURES+1))
else
  echo "PASS no raw inverse emitted"
fi
grep -q "✓ Theme → slate" "$PTY_TXT"
check "theme switch applied" $?
# slate accent is 110: the second drag follows the new theme.
grep -qF "$(printf '\x1b[38;5;16;48;5;110m')" "$PTY_TXT"
check "selection bar follows the switched theme" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT SELECT E2E: ALL PASSED"
  exit 0
fi
echo "CHAT SELECT E2E: $FAILURES FAILURE(S) — pty transcript follows"
cat "$PTY_TXT"
exit 1
