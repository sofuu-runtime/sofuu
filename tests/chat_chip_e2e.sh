#!/bin/bash
# Chip-persistence + settle-line e2e: drives ./sofuu chat on a pty against
# the mock LLM.
#   S1: small turn — the per-turn usage chip (in N · out M) appears in the
#       METRIC row and SURVIVES the 2s metricsTimer repaints (it used to
#       be wiped 2s after the turn ended, and before that truncated away
#       by the hints column on narrow terminals).
#   S2: big paste prompt — the footer's live meter climbs high during the
#       turn, then settles much lower at turn end; the settle line
#       ("ctx meter settled N→M") must explain the drop.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
PORT=18799
TMPD="$(mktemp -d)"
trap 'kill "$MOCK_PID" 2>/dev/null; [ "${SOFUU_KEEP_DBG:-}" = 1 ] || rm -rf "$TMPD"' EXIT

mkdir -p "$TMPD/.sofuu"
printf '{ "provider": "custom", "model": "mock", "base_url": "http://127.0.0.1:%s/v1/chat/completions", "api_key": "x", "brain": false, "ghost": false, "ctx_window": 32768 }\n' "$PORT" > "$TMPD/.sofuu/config.json"

"$SOFUU" run "$ROOT/tests/mock_llm_server.js" "$PORT" > "$TMPD/mock.log" 2>&1 &
MOCK_PID=$!
# 20s cap: a cold `sofuu run` boot of the mock can exceed 5s on a busy machine
for _ in $(seq 1 200); do grep -q "MOCK-LLM-READY" "$TMPD/mock.log" 2>/dev/null && break; sleep 0.1; done
grep -q "MOCK-LLM-READY" "$TMPD/mock.log" || { echo "FAIL mock not ready"; exit 1; }

PASTE_BYTES=0
HOME="$TMPD" python3 - "$TMPD" "$SOFUU" <<'PYEOF'
import os, pty, sys, time, select, signal, re

tmpd, bin_path = sys.argv[1], sys.argv[2]
os.chdir(tmpd)
pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = tmpd
    os.environ["SOFUU_HOME"] = tmpd
    os.execv(bin_path, [bin_path, "chat"])

def drain(sec):
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

def footer_metric(buf):
    """Latest 'ctx N/...' + chip paint from the raw stream."""
    s = buf.decode("utf-8", "replace")
    m = re.findall(r"ctx (\d+(?:\.\d+)?k?)/\S+(?:\x1b\[0m)?(?:\x1b\[2m · in (\d+(?:\.\d+)?k?) · out (\d+(?:\.\d+)?k?))?", s)
    return s

# ── S1: small turn ───────────────────────────────────────────────
drain(2.0)
os.write(fd, b"hello world turn one\r")
time.sleep(3.0)
snap1 = drain(0.5)
time.sleep(4.5)   # metricsTimer fires ~2x in this window
snap2 = drain(0.5)

s1 = snap1.decode("utf-8", "replace")
s2 = snap2.decode("utf-8", "replace")
chip_re = re.compile(r"in \d+(?:\.\d+)?k? · out \d+(?:\.\d+)?k?")
chip_now = "yes" if chip_re.search(s1) else "no"
chip_later = "yes" if chip_re.search(s2) else "no"
settle_small = "yes" if "ctx meter settled" in (s1 + s2) else "no"
print("S1_CHIP_AFTER_TURN=" + chip_now)
print("S1_CHIP_4S_LATER=" + chip_later)
print("S1_SETTLE_LINE(should_be_no)=" + settle_small)

# ── S2: big paste prompt — live meter high, settle drops ─────────
# Write the paste in chunks WHILE draining: a single 4KB+ os.write into
# the pty master blocks forever once the kernel buffer fills (the paste
# e2e learned this — send + drain must interleave).
lines = []
for i in range(60):
    lines.append("PASTE-SETTLE-LINE-%03d %s" % (i + 1, "p" * 60))
paste = "\n".join(lines)
payload = ("\x1b[200~" + paste + "\x1b[201~").encode()
CHUNK = 512
sent = 0
end = time.time() + 10
while sent < len(payload) and time.time() < end:
    n = os.write(fd, payload[sent:sent + CHUNK])
    sent += n
    # drain alongside so the kernel pty buffer never fills
    r, _, _ = select.select([fd], [], [], 0.02)
    if r:
        try:
            os.read(fd, 65536)
        except OSError:
            break
time.sleep(1.0)
os.write(fd, b"\r")
# 4s of timer fires guarantees ≥1 fresh repaint carrying the new chip.
# POLL until the last footer paint reflects the paste request's bigger
# prompt (cap 20s): the old fixed 4s+2.5s window lost to busy machines —
# the mock answer arrived late, the freshest paint was still turn-1's
# ~153tk, and the check read a stale chip rather than a real bug.
chip3_re = re.compile(r"in (\d+(?:\.\d+)?k?) · out \d+(?:\.\d+)?k?")
# the paste request's real prompt tokens (mock sizes the request) must
# appear in the chip: ~4.3k user bytes/4 ≈ 1100+ tk, far above turn-1's
# ~100 — fmtTk renders it as "1.3k", so parse the k-suffix.
def tk_num(tok):
    return int(float(tok[:-1]) * 1000) if tok.endswith("k") else int(float(tok))
s3 = ""
deadline = time.time() + 20.0
while time.time() < deadline:
    s3 += drain(1.5).decode("utf-8", "replace")
    paints = chip3_re.findall(s3)
    if paints and tk_num(paints[-1]) > 1000:
        break

settle_big = "yes" if "ctx meter settled" in s3 else "no"
# A paste GROWS history — the meter settles HIGHER (not a drop), so the
# settle line must NOT fire here either. The tool-transcript drop it
# explains can't be scripted with this stateless mock (no tool_calls);
# the line is defensive UX for real tool turns.
print("S2_SETTLE_LINE(paste settles higher, should_be_no)=" + settle_big)
# The raw stream carries EVERY footer paint, oldest first — take the
# LAST chip paint (the freshest footer), not the first stale one.
paints = chip3_re.findall(s3)
chip3 = paints[-1] if paints else "0"
print("S2_CHIP_IN_TK=" + chip3)
big = tk_num(chip3) > 1000
print("S2_CHIP_REFLECTS_PASTE(>1000)=" + ("yes" if big else "no"))

try: os.kill(pid, signal.SIGKILL)
except Exception: pass
try: os.waitpid(pid, 0)
except Exception: pass
os.close(fd)

ok = (chip_now == "yes" and chip_later == "yes" and settle_small == "no"
      and settle_big == "no" and big)
sys.exit(0 if ok else 1)
PYEOF
RC=$?
exit $RC
