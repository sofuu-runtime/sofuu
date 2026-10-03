#!/usr/bin/env bash
# tests/chat_sessions_e2e.sh — the /sessions list must be keyboard-usable.
#
# Regression: /sessions printed a static table (session::cmd_list) that
# looked selectable but took no keys — arrows did nothing and Enter did
# nothing. Bare /sessions must now open the same keyboard-driven picker as
# /resume (↑↓ moves, Enter resumes), titled with the project it is scoped
# to. This drives the REAL chat on a pty against a fixture project with
# two registry sessions and asserts:
#   1. the picker opens, titled "Sessions for <proj>" (dir scoping),
#   2. both fixture sessions are listed,
#   3. ↓ moves the ❯ marker onto the second row,
#   4. Enter resumes the highlighted session (it has no turns, so the
#      resume path reports that instead of silently doing nothing).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"

FAILURES=0
check() { # check <name> <cond-exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILURES=$((FAILURES+1)); fi
}

TMP="$(mktemp -d /tmp/sofuu_chat_sessions_XXXXXX)"
HOME_P="$TMP/home"
PROJ="$TMP/proj"
mkdir -p "$HOME_P/.sofuu" "$PROJ/.sofuu/sessions"
trap 'rm -rf "$TMP"' EXIT

cat > "$HOME_P/.sofuu/config.json" <<EOF
{ "provider": "custom", "model": "mock-model",
  "base_url": "http://127.0.0.1:9/v1/chat/completions",
  "api_key": "x", "brain": false, "ghost": false, "effort": "high" }
EOF

# Two ended sessions with distinct tasks and no stores on disk (0 turns).
# Full SessionInfo shape — a missing field fails the whole registry parse
# (read() fails open to EMPTY), which is exactly what this must not do.
cat > "$PROJ/.sofuu/sessions/registry.json" <<EOF
{ "sessions": [
  { "id": "s-abcdef1234567-abcde-1234", "pid": 1, "host": "t",
    "cwd": "$PROJ", "model": "live-alpha", "provider": "custom",
    "started_at": 1790300000, "last_seen": 1790300000,
    "task": "alpha-task", "ended": true },
  { "id": "s-1234567890abc-def12-3456", "pid": 2, "host": "t",
    "cwd": "$PROJ", "model": "live-alpha", "provider": "custom",
    "started_at": 1790300100, "last_seen": 1790300100,
    "task": "beta-task", "ended": true }
] }
EOF

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
sent_sessions = sent_down = sent_enter = False
while time.time() - start < 20:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    # 1) open the sessions browser
    if not sent_sessions and time.time() - start > 5:
        sent_sessions = True
        os.write(fd, b"/sessions\r")
    # 2) move down one row
    if sent_sessions and not sent_down and time.time() - start > 7.5:
        sent_down = True
        os.write(fd, b"\x1b[B")
    # 3) enter resumes the highlighted (second) session
    if sent_down and not sent_enter and time.time() - start > 9.5:
        sent_enter = True
        os.write(fd, b"\r")
    if time.time() - start > 12:
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
# The picker's selection marker is ESC[35m❯ (magenta) + bold; the input
# prompt is ESC[1;35m❯ — so `\[35m❯` matches ONLY a real picker row, never
# the input box. (A bare `❯` grep passed even on the old static table.)
grep -q "Sessions for $PROJ" "$PTY_TXT"
check "picker opens titled with the project it is scoped to" $?
grep -q "alpha-task" "$PTY_TXT"
check "first fixture session is listed" $?
grep -q "beta-task" "$PTY_TXT"
check "second fixture session is listed" $?
grep -q '\[35m❯' "$PTY_TXT"
check "keyboard selection marker renders (a picker, not a static table)" $?
grep -q '\[35m❯.\{0,80\}beta-task' "$PTY_TXT"
check "↓ moved the marker onto the second row" $?
grep -q "Session has no turns to resume" "$PTY_TXT"
check "Enter resumed the highlighted session (resume path, not silence)" $?

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CHAT SESSIONS E2E: ALL PASSED"
  exit 0
fi
echo "CHAT SESSIONS E2E: $FAILURES FAILURE(S) — pty transcript follows"
cat "$PTY_TXT"
exit 1
