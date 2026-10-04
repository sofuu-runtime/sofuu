#!/usr/bin/env bash
# tests/chat_transcript_e2e.sh — transcript hierarchy, rhythm, checklist.
#
# The transcript used to render as one undifferentiated white blob: answers
# showed raw markdown markers, tool groups ran into the prose above them,
# and todo_write printed only "checklist updated (N steps)" — the live
# checklist the event already carried was never painted. This drives the
# REAL chat on a pty (mock answers markdown, then calls todo_write) and
# asserts:
#   1. headings/bold/code render with their levels (bold-cyan / bold /
#      yellow), not as raw markers,
#   2. a blank row opens the tool group (whitespace rhythm),
#   3. the todo checklist paints (☑ header + ✓/▸/○ rows) INSTEAD of the
#      one-row summary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29600 + RANDOM % 500))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_transcript_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# ── mock LLM: todo_write call first, markdown answer second ─────────
# Quoted heredoc (nothing expands — the md string holds backticks), then
# the port token is substituted.
cat > "$TMP/mock.js" <<'MOCKEOF'
let n = 0;
const server = sofuu.http.createServer(function (req, res) {
  if (String(req.url || "").indexOf("/models") >= 0) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ object: "list", data: [{ id: "live-alpha", context_length: 1048576, max_completion_tokens: 65536 }] }));
    return;
  }
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  let body = {}; try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  const hasTools = !!(body.tools && body.tools.length);
  const C = (d, f) => ({ id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: d, finish_reason: f }] });
  const W = (o) => res.write("data: " + JSON.stringify(o) + "\n\n");
  n++;
  if (hasTools && n === 1) {
    const a = JSON.stringify({ todos: [
      { content: "first step", status: "done" },
      { content: "second step", status: "doing" },
      { content: "third step", status: "todo" } ] });
    W(C({ tool_calls: [{ index: 0, id: "c1", type: "function",
      function: { name: "todo_write", arguments: a } }] }, null));
    W(C({}, "tool_calls"));
  } else {
    const md = "## Work Plan\n\nDone **fast** with `code`.\n\n> quoted note\n\n---\n\ntail";
    W(C({ content: md }, null));
    W(C({}, "stop"));
  }
  res.write("data: [DONE]\n\n"); res.end();
});
server.listen(__TX_PORT__, "127.0.0.1");
console.log("MOCK-TX-READY");
MOCKEOF
sed -i '' "s/__TX_PORT__/$PORT/" "$TMP/mock.js" 2>/dev/null || sed -i "s/__TX_PORT__/$PORT/" "$TMP/mock.js"
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run "$TMP/mock.js" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-TX-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-TX-READY" "$TMP/mock.log" || { echo "mock failed to start"; cat "$TMP/mock.log"; exit 1; }

# ── drive the chat TUI on a pty ──────────────────────────────────
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
sent = done = False
while time.time() - start < 25:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    if not sent and time.time() - start > 6:
        sent = True
        os.write(fd, b"do the thing\r")
    if sent and not done and time.time() - start > 18:
        done = True
        os.write(fd, b"/exit\r")
    if time.time() - start > 21:
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

# ── assertions (raw stream: escapes intact) ──────────────────────
# 1. markdown hierarchy: levels render, markers consumed by the format.
grep -qF "$(printf '\x1b[1;36mWork Plan')" "$PTY_TXT"
check "heading renders bold-cyan" $?
grep -qF "$(printf '\x1b[1mfast\x1b[0m')" "$PTY_TXT"
check "bold renders bold" $?
grep -qF "$(printf '\x1b[33mcode\x1b[0m')" "$PTY_TXT"
check "inline code renders yellow" $?
# 2. whitespace rhythm: a painted blank row (gutter + single space — bare
# clears carry no bytes after K, and every content row starts with the
# 2-space gutter, so `K + 3 spaces + ESC` matches ONLY a real blank row)
# opens the dim-⏺ tool line. (The input prompt is bold-magenta ⏺, and
# panel blanks are bare clears — neither can match.)
grep -q "$(printf '\x1b\\[K   \x1b\\[[0-9]*;1H\x1b\\[K  \x1b\\[90m  \xe2\x8f\xba')" "$PTY_TXT"
check "blank row opens the tool group" $?
# 3. live checklist instead of the one-row summary.
grep -q "☑ 1/3" "$PTY_TXT"
check "checklist header with progress" $?
grep -q "✓ first step" "$PTY_TXT"
check "done step renders ✓" $?
grep -q "▸ second step" "$PTY_TXT"
check "doing step renders ▸" $?
grep -q "○ third step" "$PTY_TXT"
check "queued step renders ○" $?
! grep -q "checklist updated (3 steps)" "$PTY_TXT"
check "one-row summary replaced by the list, not duplicated" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT TRANSCRIPT E2E: ALL PASSED"
  exit 0
fi
echo "CHAT TRANSCRIPT E2E: $FAILURES FAILURE(S) — pty transcript follows"
cat "$PTY_TXT"
exit 1
