#!/usr/bin/env bash
# tests/chat_compact_role_e2e.sh — compaction must survive strict gateways.
#
# Regression (2026-10-09, live session): every auto + manual compaction on
# tokenrouter died with `[Compact failed: ... "The last message must have
# role=user."]`. summarizeHistory sent [{system}, ...foldedPrefix] and the
# folded prefix ends with an assistant message — strict OpenAI-wire
# gateways 400 that shape. The instruction now rides LAST as a user
# message, so the payload always ends with role=user.
#
# Drives the REAL chat on a pty: two mock turns, then /compact, against an
# inline mock that 400s exactly like the gateway when the last message is
# not role=user. Old code prints [Compact failed: ...role=user...]; fixed
# code prints [Compacted → ...].
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29900 + RANDOM % 400))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_role_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# ── mock LLM: strict last-role=user gate on summarizations ──────
cat > "$TMP/mock.js" <<'MOCKEOF'
const server = sofuu.http.createServer(function (req, res) {
  if (String(req.url || "").indexOf("/models") >= 0) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ object: "list", data: [{ id: "live-alpha", context_length: 1048576, max_completion_tokens: 65536 }] }));
    return;
  }
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  const sys = msgs.length ? String((msgs[0] && msgs[0].content) || "") : "";
  const lastRole = msgs.length ? String(msgs[msgs.length - 1].role || "") : "";
  const isSummarize = sys.toLowerCase().indexOf("summariz") >= 0;
  if (isSummarize && lastRole !== "user") {
    /* the live gateway's exact rejection (2026-10-09 tokenrouter) */
    res.writeHead(400, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ error: { message: "The last message must have role=user.", type: "invalid_request_error", code: "invalid_request_error" } }));
    return;
  }
  if (isSummarize) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ choices: [{ message: { role: "assistant", content: "SUMMARY-OK: user discussed things." } }],
      usage: { prompt_tokens: 10, completion_tokens: 5 } }));
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
console.log("MOCK-ROLE-READY");
MOCKEOF
sed -i '' "s/__TX_PORT__/$PORT/" "$TMP/mock.js" 2>/dev/null || sed -i "s/__TX_PORT__/$PORT/" "$TMP/mock.js"
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run "$TMP/mock.js" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-ROLE-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-ROLE-READY" "$TMP/mock.log" || { echo "mock failed to start"; cat "$TMP/mock.log"; exit 1; }

# ── pty: two turns (manual compact needs >1 block), then /compact ─
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
s1 = s2 = s3 = False
while time.time() - start < 40:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    t = time.time() - start
    if not s1 and t > 6:
        s1 = True
        os.write(fd, b"first thing\r")
    if s1 and not s2 and t > 14:
        s2 = True
        os.write(fd, b"second thing\r")
    if s2 and not s3 and t > 24:
        s3 = True
        os.write(fd, b"/compact\r")
    if t > 33:
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

# ── assertions ───────────────────────────────────────────────────
grep -q "TURN-DONE" "$PTY_TXT"
check "both mock turns completed" $?
if grep -q "Compact failed" "$PTY_TXT"; then
  echo "FAIL /compact returned Compact failed (last message was not role=user)"
  FAILURES=$((FAILURES+1))
else
  echo "PASS /compact did not fail"
fi
grep -q "Compacted" "$PTY_TXT"
check "/compact folded history into a summary" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT COMPACT ROLE E2E: ALL PASSED"
  exit 0
fi
echo "CHAT COMPACT ROLE E2E: $FAILURES FAILURE(S) — pty transcript follows"
cat "$PTY_TXT"
exit 1
