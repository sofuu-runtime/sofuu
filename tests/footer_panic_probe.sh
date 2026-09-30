#!/bin/bash
# Footer-metric panic probe (2026-09-09 crash: process.rs:1366 "range end
# index 141 out of range for slice of length 80").
#
# Reproduces the scenario: real chat turns on a WIDE terminal (default
# 157x38 = the crash geometry) so the ctx meter (METRIC_LEFT) populates
# and the footer repaints after each answer. With the strlen-walking
# truncator the repaint can read past the non-NUL-terminated metric
# buffer into adjacent heap and slice out of range; panic=abort then
# kills the whole CLI.
#
# Uses the mock LLM only (no live keys, throwaway HOME).
#
# HEAP-LAYOUT DEPENDENT: a clean run is NOT proof of safety — the
# deterministic guarantee is the bounded truncator + its unit tests.
# This probe is best-effort live evidence: CRASH (exit 2) = regression,
# OK (exit 0) = no panic observed.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFUU="$ROOT/sofuu"
# Outside every contested test range (18699-18999 draws, 19803 modes probe).
PORT=19811
COLS="${1:-157}"
ROWS="${2:-38}"
TMPD="$(mktemp -d)"
trap 'kill "$MOCK_PID" 2>/dev/null; [ "${SOFUU_KEEP_DBG:-}" = 1 ] || rm -rf "$TMPD"' EXIT

mkdir -p "$TMPD/.sofuu"
printf '{ "provider": "custom", "model": "mock", "base_url": "http://127.0.0.1:%s/v1/chat/completions", "api_key": "x", "brain": false, "ghost": false, "ctx_window": 32768 }\n' "$PORT" > "$TMPD/.sofuu/config.json"

"$SOFUU" run "$ROOT/tests/mock_llm_server.js" "$PORT" > "$TMPD/mock.log" 2>&1 &
MOCK_PID=$!
# 20s cap: a cold `sofuu run` boot of the mock can exceed 5s on a busy machine
for _ in $(seq 1 200); do grep -q "MOCK-LLM-READY" "$TMPD/mock.log" 2>/dev/null && break; sleep 0.1; done
grep -q "MOCK-LLM-READY" "$TMPD/mock.log" || { echo "FAIL mock not ready"; exit 1; }

HOME="$TMPD" python3 - "$TMPD" "$SOFUU" "$COLS" "$ROWS" <<'PYEOF'
import os, pty, sys, time, select, re, fcntl, termios, struct

tmpd, bin_path, cols, rows = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.chdir(tmpd)

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = tmpd
    os.environ["SOFUU_HOME"] = tmpd
    os.execv(bin_path, [bin_path, "chat"])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

def drain(quiet=0.8, deadline=20.0):
    # Read until the stream goes quiet. A fixed sleep slices the stream
    # mid-repaint; footer repaints land in bursts after each turn.
    buf = b""
    end = time.time() + deadline
    last = time.time()
    while time.time() < end and time.time() - last < quiet:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                break
            if not d:
                break
            buf += d
            last = time.time()
    return buf

def send(line):
    os.write(fd, (line + "\r").encode())

time.sleep(1.0)
drain(quiet=0.8, deadline=10.0)  # boot + welcome panel

crash = b""
prompts = [
    "Explain in detail how a static linker resolves symbols, at least 300 words.",
    "Now do the same for dynamic linking, another 300 words.",
    "Compare the two approaches in 300 more words.",
]
for p in prompts:
    send(p)
    out = drain(quiet=1.5, deadline=30.0)
    if b"panicked at" in out:
        crash = out
        break
    # idle window: the periodic footer/metric repaint fires here
    out2 = drain(quiet=1.0, deadline=5.0)
    if b"panicked at" in out2:
        crash = out2
        break

if not crash:
    send("/quit")
    tail = drain(quiet=0.5, deadline=5.0)
    if b"panicked at" in tail:
        crash = tail

if crash:
    m = re.search(rb"panicked at[^\r\n]*", crash)
    print("CRASH REPRODUCED:", m.group(0).decode(errors="replace") if m else "panic present")
    sys.exit(2)

print("OK: no panic across 3 turns at %dx%d (heap-layout dependent; the bounded-fn unit tests are the deterministic gate)" % (cols, rows))
sys.exit(0)
PYEOF
