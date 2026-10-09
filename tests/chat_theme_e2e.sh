#!/usr/bin/env bash
# tests/chat_theme_e2e.sh — 25 user-selectable TUI themes.
#
# Themes are fixed 256-palette colors (never the terminal-themed 30-37
# range), so they render identically under any terminal theme. This drives
# the REAL chat on a pty plus two piped runs and asserts:
#   1. /theme opens a picker listing dark + light themes,
#   2. /theme slate applies it (confirmation names it, panel border and
#      later tool lines repaint in slate colors — the switch is live,
#      not next-session),
#   3. the choice persists in config.json,
#   4. piped /theme lists all 25 and rejects unknown names without
#      touching the active theme.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT="${1:-$((29700 + RANDOM % 500))}"
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_theme_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ"
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model", "base_url": "$BASE",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# ── mock LLM: models endpoint + one todo_write turn ──────────────
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
    const a = JSON.stringify({ todos: [{ content: "only step", status: "doing" }] });
    W(C({ tool_calls: [{ index: 0, id: "c1", type: "function",
      function: { name: "todo_write", arguments: a } }] }, null));
    W(C({}, "tool_calls"));
  } else {
    W(C({ content: "THEME-TURN-DONE" }, null));
    W(C({}, "stop"));
  }
  res.write("data: [DONE]\n\n"); res.end();
});
server.listen(__TX_PORT__, "127.0.0.1");
console.log("MOCK-THEME-READY");
MOCKEOF
sed -i '' "s/__TX_PORT__/$PORT/" "$TMP/mock.js" 2>/dev/null || sed -i "s/__TX_PORT__/$PORT/" "$TMP/mock.js"
( cd "$ROOT" && exec env HOME="$HOME_P" "$SOFUU" run "$TMP/mock.js" ) > "$TMP/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 200); do
  grep -q "MOCK-THEME-READY" "$TMP/mock.log" 2>/dev/null && break
  sleep 0.1
done
grep -q "MOCK-THEME-READY" "$TMP/mock.log" || { echo "mock failed to start"; cat "$TMP/mock.log"; exit 1; }

# ── pty: open picker, apply slate by name, run a turn ───────────
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
s1 = s2 = s3 = s4 = s5 = s6 = s7 = False
while time.time() - start < 36:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    # 1) open the theme picker, filter to light themes (they sit below
    # the 10-row fold unfiltered), then cancel — listing is the assert
    if not s1 and time.time() - start > 6:
        s1 = True
        os.write(fd, b"/theme\r")
    if s1 and not s2 and time.time() - start > 8.5:
        s2 = True
        os.write(fd, b"light")
    if s2 and not s3 and time.time() - start > 10.5:
        s3 = True
        os.write(fd, b"\x1b")
    # 2) apply slate directly
    if s3 and not s4 and time.time() - start > 12:
        s4 = True
        os.write(fd, b"/theme slate\r")
    # 3) a turn AFTER the switch — its tool line must wear slate
    if s4 and not s5 and time.time() - start > 14.5:
        s5 = True
        os.write(fd, b"do the thing\r")
    # 4) switch to a LIGHT theme mid-session — the whole window must
    # flip background (not just future text) and old rows remap.
    if s5 and not s6 and time.time() - start > 19:
        s6 = True
        os.write(fd, b"/theme paper\r")
    # 5) clean exit INSIDE the read loop so the exit bytes (SGR reset +
    # alt-screen leave) land in the transcript for the no-leak assert.
    if s6 and not s7 and time.time() - start > 24:
        s7 = True
        os.write(fd, b"/exit\r")
    if time.time() - start > 32:
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

# ── assertions: picker + live switch + persistence ───────────────
grep -q "Theme" "$PTY_TXT"
check "theme picker opens" $?
# slate with its "dark theme" note: listing context, not a stray substring.
grep -q "slate.*dark theme" "$PTY_TXT"
check "picker lists slate" $?
# "light" filters to exactly the 12 light themes; paper is first and
# always above the fold (moss-light sits 11th — asserting it would pin
# scroll position, not listing).
grep -q "paper" "$PTY_TXT"
check "filtering to light themes lists paper" $?
grep -q "12 matches" "$PTY_TXT"
check "the light set is exactly the 12 light themes" $?
grep -q "✓ Theme → slate" "$PTY_TXT"
check "applying by name confirms (dark, position)" $?
# slate tool color 38;5;109 on the post-switch turn (default is 38;5;80).
grep -qF "$(printf '\x1b[38;5;109mtodo_write')" "$PTY_TXT"
check "post-switch tool line wears the slate tool color" $?
# slate panel border 2;38;5;110 repainted by the refresh.
grep -qF "$(printf '\x1b[2;38;5;110m')" "$PTY_TXT"
check "panel border repaints in the slate color" $?
# slate is dark: every erase fills with its window bg 235.
grep -qF "$(printf '\x1b[48;5;235m')" "$PTY_TXT"
check "dark theme paints the window background (48;5;235)" $?
grep -q "✓ Theme → paper" "$PTY_TXT"
check "mid-session switch to a light theme confirms" $?
# paper is light: the whole window flips to bg 255…
grep -qF "$(printf '\x1b[48;5;255m')" "$PTY_TXT"
check "light theme flips the window background (48;5;255)" $?
# …and the slate tool line from the earlier turn repaints remapped to
# the paper tool color (old rows flip foregrounds with the window —
# a polarity switch must not strand light text on a light window).
grep -qF "$(printf '\x1b[38;5;29mtodo_write')" "$PTY_TXT"
check "old rows remap foregrounds on the switch" $?
# leaving the TUI resets SGR before dropping the alt screen, so the
# themed background never leaks into the user's shell.
grep -qF "$(printf '\x1b[0m\x1b[?25h\x1b[?1049l')" "$PTY_TXT"
check "exit resets colors before leaving the alt screen" $?
grep -q '"theme": *"paper"' "$HOME_P/.sofuu/config.json"
check "choice persists in config.json" $?

# ── piped: list + unknown-name rejection ─────────────────────────
printf '/theme\n/exit\n' |
  HOME="$HOME_P" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/piped_list" 2>&1
sed -i '' $'s/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/piped_list" 2>/dev/null || sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$TMP/piped_list"
NTHEMES="$(grep -cE '^  [a-z0-9-]+  +(dark|light)' "$TMP/piped_list" || true)"
[ "$NTHEMES" -eq 25 ]
check "piped /theme lists all 25 themes (got $NTHEMES)" $?
printf '/theme nope-nope\n/exit\n' |
  HOME="$HOME_P" SOFUU_PROJECT="$PROJ" "$SOFUU" chat >"$TMP/piped_bad" 2>&1
grep -q "Unknown theme 'nope-nope'" "$TMP/piped_bad"
check "unknown name rejected with a hint, not applied" $?
grep -q '"theme": *"paper"' "$HOME_P/.sofuu/config.json"
check "rejected name did not clobber the active theme" $?
# window backgrounds are a TTY paint concern — piped output must stay
# byte-clean of them (both polarities).
if grep -q '48;5;25[55]' "$TMP/piped_list" "$TMP/piped_bad"; then
  echo "FAIL themed background leaked into piped output"
  FAILURES=$((FAILURES+1))
else
  echo "PASS no themed background in piped output"
fi

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT THEME E2E: ALL PASSED"
  exit 0
fi
echo "CHAT THEME E2E: $FAILURES FAILURE(S) — transcripts follow"
echo "===== pty ====="; cat "$PTY_TXT"
echo "===== piped list ====="; cat "$TMP/piped_list"
echo "===== piped bad ====="; cat "$TMP/piped_bad"
exit 1
