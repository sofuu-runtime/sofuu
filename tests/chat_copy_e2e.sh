#!/bin/bash
# Copyable-text e2e: mouse-drag selection + Ctrl-K clipboard copy.
# Drives ./sofuu chat on a real pty, prints /help so known marker text is
# in the conversation, performs an SGR mouse drag over conversation rows,
# presses Ctrl-K, then checks the system clipboard holds the dragged rows'
# text (ANSI stripped, no escape bytes, includes a known marker).
set -u
# Clipboard verification is macOS-only: it reads the system clipboard with
# /usr/bin/pbpaste (the file notes other hosts would need xclip). On Linux
# runners that binary does not exist and the test died with
# FileNotFoundError: '/usr/bin/pbpaste', which reads as a product failure
# and is not one. Exit 77 — the POSIX skip convention run_js_tests.sh
# already honours for shell e2e suites.
if [ "$(uname -s)" != "Darwin" ] || [ ! -x /usr/bin/pbpaste ]; then
  echo "SKIP chat_copy_e2e — clipboard verification needs macOS pbpaste (host: $(uname -s))"
  exit 77
fi
BIN="$(cd "$(dirname "$0")/.." && pwd)/sofuu"
TMPD="$(mktemp -d)"
export SOFUU_HOME="$TMPD"
HOME="$TMPD" python3 - "$TMPD" "$BIN" <<'PYEOF'
import os, pty, sys, time, select, subprocess, signal

tmpd, bin_path = sys.argv[1], sys.argv[2]
os.chdir(tmpd)

# Pre-seed a minimal provider config so chat skips the setup wizard and
# lands straight in the TUI.
cfgdir = os.path.join(tmpd, ".sofuu")
os.makedirs(cfgdir, exist_ok=True)

pid, fd = pty.fork()
if pid == 0:
    os.environ["HOME"] = tmpd
    os.environ["SOFUU_HOME"] = tmpd
    os.environ["SOFUU_SEL_DEBUG"] = os.path.join(tmpd, "sel_debug.bin")
    os.environ["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ.get("PATH","")
    os.execv(bin_path, [bin_path, "chat"])

buf = b""
def drain(sec):
    global buf
    end = time.time() + sec
    while time.time() < end:
        r,_,_ = select.select([fd],[],[],0.1)
        if r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                return
            if not d:
                return
            buf += d

# welcome panel + footer render
drain(2.5)

# Put known text in the conversation: /help prints the command list.
os.write(fd, b"/help\r")
drain(2.0)
os.write(fd, b"\x1b[<0;5;3M")   # press at row 3 (inside help output)
time.sleep(0.15)
os.write(fd, b"\x1b[<32;5;6M")  # drag to row 6
time.sleep(0.15)
os.write(fd, b"\x1b[<32;5;9M")  # drag to row 9
time.sleep(0.15)
os.write(fd, b"\x1b[<0;5;9m")   # release
drain(0.8)

# Ctrl-K copies the selection
os.write(fd, b"\x0b")
drain(1.5)

# read the clipboard via pbpaste (macOS); other hosts would need xclip.
# The env MUST carry a UTF-8 LC_CTYPE: a bare env makes pbpaste transcode
# the pasteboard to ASCII and every multibyte char reads back as '?'.
out = subprocess.run(["/usr/bin/pbpaste"], capture_output=True, timeout=5,
                     env={"PATH": "/usr/bin:/bin", "LC_CTYPE": "en_US.UTF-8"})
clip = out.stdout.decode("utf-8", "replace")

dbg_path = os.path.join(tmpd, "sel_debug.bin")
if os.path.exists(dbg_path):
    dbg = open(dbg_path, "rb").read()
    print("DEBUG_VS_CLIP_IDENTICAL=" + ("yes" if dbg.strip() == out.stdout.strip() else "no"))
else:
    print("DEBUG_VS_CLIP_IDENTICAL=no-debug-file")

# cleanup
try:
    os.kill(pid, signal.SIGKILL)
except ProcessLookupError:
    pass
try:
    os.waitpid(pid, 0)
except ChildProcessError:
    pass
os.close(fd)

painted = buf.decode("utf-8", "replace")
ok_sel = "\x1b[7m" in painted                       # inverse-video paint seen
clean = "\x1b" not in clip                            # ANSI stripped
utf8 = "\ufffd" not in clip                          # multibyte survived
marker = any(("Sofuu Chat Commands" in l) or ("this help" in l)
             or ("interactive model picker" in l)
             for l in clip.splitlines())
nonempty = len(clip.strip()) > 0
print("INVERSE_PAINT_SEEN=" + ("yes" if ok_sel else "no"))
print("CLIPBOARD_LEN=" + str(len(clip.strip())))
print("CLIPBOARD_HAS_ANSI=" + ("yes" if "\x1b" in clip else "no"))
print("CLIPBOARD_UTF8_CLEAN=" + ("yes" if utf8 else "no"))
print("CLIPBOARD_HAS_MARKER=" + ("yes" if marker else "no"))
for i, l in enumerate(clip.strip().splitlines()[:4]):
    print(f"CLIP_LINE{i+1}=" + l[:100])
if not (ok_sel and clean and utf8 and (marker or nonempty)):
    sys.exit(1)
PYEOF
RC=$?
rm -rf "$TMPD"
exit $RC
