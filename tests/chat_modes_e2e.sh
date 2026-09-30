#!/bin/bash
# Permission-modes e2e: drives ./sofuu chat on a pty against the mock LLM.
#   W1: welcome panel shows the Mode row (default "full access").
#   S1: /mode bare prints the current mode + usage.
#   S2: /plan echoes the switch, re-renders the welcome row, and repaints
#       the footer with the amber "mode plan" chip.
#   S3: /mode edit sets edit; /mode sudo is refused and keeps the prior mode.
#   S4: /full restores full access.
#   S6: TAB on a plain line cycles full → edit → plan → full (echo, chip,
#       welcome refresh, immediate persistence), and TAB after '/' is still
#       slash-completion — the two share the key without collision.
#   S5: cfg.permissions PERSISTS into config.json across an exit/restart, and
#       the restarted session boots straight into plan (welcome row + chip).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
# Fixed port OUTSIDE every random range in tests/ (18699-18999 is contested
# by brain_auth/no_response/paste draws) — a held port here fails the boot.
PORT=19803
TMPD="$(mktemp -d)"
trap 'kill "$MOCK_PID" 2>/dev/null; [ "${SOFUU_KEEP_DBG:-}" = 1 ] || rm -rf "$TMPD"' EXIT

mkdir -p "$TMPD/.sofuu"
printf '{ "provider": "custom", "model": "mock", "base_url": "http://127.0.0.1:%s/v1/chat/completions", "api_key": "x", "brain": false, "ghost": false, "ctx_window": 32768 }\n' "$PORT" > "$TMPD/.sofuu/config.json"

"$SOFUU" run "$ROOT/tests/mock_llm_server.js" "$PORT" > "$TMPD/mock.log" 2>&1 &
MOCK_PID=$!
# 20s cap: a cold `sofuu run` boot of the mock can exceed 5s on a busy machine
for _ in $(seq 1 200); do grep -q "MOCK-LLM-READY" "$TMPD/mock.log" 2>/dev/null && break; sleep 0.1; done
grep -q "MOCK-LLM-READY" "$TMPD/mock.log" || { echo "FAIL mock not ready"; exit 1; }

HOME="$TMPD" python3 - "$TMPD" "$SOFUU" <<'PYEOF'
import os, pty, sys, time, select, signal, json, re

tmpd, bin_path = sys.argv[1], sys.argv[2]
os.chdir(tmpd)

def spawn():
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["HOME"] = tmpd
        os.environ["SOFUU_HOME"] = tmpd
        os.execv(bin_path, [bin_path, "chat"])
    return pid, fd

def drain(fd, sec):
    buf = b""
    end = time.time() + sec
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                break
            if not d:
                break
            buf += d
    return buf

def kill(pid, fd):
    try: os.kill(pid, signal.SIGKILL)
    except Exception: pass
    try: os.waitpid(pid, 0)
    except Exception: pass
    try: os.close(fd)
    except Exception: pass

def strip(s):
    # welcome rows are dim-ANSI painted; compare against the plain text
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

ok_all = True
def expect(tag, cond):
    global ok_all
    print("%s=%s" % (tag, "yes" if cond else "no"))
    ok_all = ok_all and cond

# ── Session 1: the mode commands ─────────────────────────────────
pid, fd = spawn()
w = strip(drain(fd, 2.5).decode("utf-8", "replace"))
expect("W1_welcome_mode_row", "Mode:" in w and "full access" in w)

os.write(fd, b"/mode\r");           time.sleep(1.0)
s = strip(drain(fd, 0.5).decode("utf-8", "replace"))
expect("S1_bare_mode_shows_current", "mode: full" in s and "/mode full|edit|plan" in s)

os.write(fd, b"/plan\r");           time.sleep(1.2)
s = strip(drain(fd, 1.0).decode("utf-8", "replace"))
expect("S2_plan_echo",
       "✓ mode → plan" in s and "read-only" in s)
expect("S2_welcome_refreshes_to_plan", "plan - read-only" in s)
expect("S2_footer_chip_mode_plan", "mode plan" in s)

os.write(fd, b"/mode edit\r");      time.sleep(1.2)
s = strip(drain(fd, 1.0).decode("utf-8", "replace"))
expect("S3_edit_echo", "✓ mode → edit" in s and "local writes" in s)

os.write(fd, b"/mode sudo\r");      time.sleep(1.0)
s = strip(drain(fd, 0.5).decode("utf-8", "replace"))
expect("S3_unknown_mode_refused", "Usage: /mode" in s)
os.write(fd, b"/mode\r");           time.sleep(1.0)
s = strip(drain(fd, 0.5).decode("utf-8", "replace"))
expect("S3_refused_keeps_prior_mode", "mode: edit" in s)

os.write(fd, b"/full\r");           time.sleep(1.0)
s = strip(drain(fd, 0.5).decode("utf-8", "replace"))
expect("S4_full_echo", "✓ mode → full" in s)

# ── S6: TAB cycles the mode on non-slash lines ───────────────────
# State here is full (S4). Each press reuses the /mode path: echo + chip +
# welcome refresh. Then '/'-completion must still win on slash lines.
os.write(fd, b"\t");                time.sleep(0.8)
s = strip(drain(fd, 1.0).decode("utf-8", "replace"))
expect("S6_tab1_full_to_edit", "✓ mode → edit" in s and "local writes" in s)
expect("S6_tab1_chip_mode_edit", "mode edit" in s)

os.write(fd, b"\t");                time.sleep(0.8)
s = strip(drain(fd, 1.0).decode("utf-8", "replace"))
expect("S6_tab2_edit_to_plan", "✓ mode → plan" in s and "plan - read-only" in s)
try:
    mid = json.load(open(os.path.join(tmpd, ".sofuu", "config.json")))
except Exception:
    mid = {}
expect("S6_tab_persists_immediately", mid.get("permissions") == "plan")

# TAB after '/' is completion, not cycling. "/pl" uniquely matches /plan
# (/mode can't probe this: /model shares every "/mode" prefix), so TAB must
# expand the line to "/plan " — if cycling had preempted it, the line would
# still read "/pl" and the mode would have moved to full instead.
os.write(fd, b"/pl");               time.sleep(0.4)
os.write(fd, b"\t");                time.sleep(0.6)
s = strip(drain(fd, 0.6).decode("utf-8", "replace"))
expect("S6_completion_not_preempted", "/plan " in s)
os.write(fd, b"\r");                time.sleep(0.8)
s = strip(drain(fd, 0.5).decode("utf-8", "replace"))
expect("S6_completion_state_consistent", "✓ mode → plan" in s)

# One more TAB returns to full — and the amber chip must disappear.
os.write(fd, b"\t");                time.sleep(0.8)
s = strip(drain(fd, 1.0).decode("utf-8", "replace"))
expect("S6_tab3_plan_to_full", "✓ mode → full" in s)
expect("S6_no_chip_at_full", "mode full" not in s)

# Restart straight into plan to prove persistence + boot-apply.
os.write(fd, b"/plan\r");           time.sleep(1.2)
drain(fd, 0.5)
kill(pid, fd)

try:
    saved = json.load(open(os.path.join(tmpd, ".sofuu", "config.json")))
except Exception as e:
    saved = {}
    print("S5_config_read_error=%r" % (e,))
expect("S5_permissions_persisted", saved.get("permissions") == "plan")

pid, fd = spawn()
w = strip(drain(fd, 2.5).decode("utf-8", "replace"))
expect("S5_restart_boots_into_plan_welcome", "plan - read-only" in w)
expect("S5_restart_boots_into_plan_chip", "mode plan" in w)
kill(pid, fd)

sys.exit(0 if ok_all else 1)
PYEOF
RC=$?
exit $RC
