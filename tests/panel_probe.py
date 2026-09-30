#!/usr/bin/env python3
"""Welcome-panel rectangle probe: reconstruct the rendered screen and verify
the box is a closed, aligned rectangle at several terminal sizes.

Spawns `sofuu chat` on a pty (TIOCSWINSZ sets the size), drains the boot
stream, replays it through a minimal terminal model (CUP / EL / 2J / SGR /
printables), then asserts every panel box row shares the same left border
column, right border column, and width — i.e. no row was torn by soft-wrap.
The walk scopes to the run of box rows around the panel title so the input
box's own borders at the bottom of the screen are excluded.
No files are written: the probe runs with a throwaway HOME and default config.
"""
import fcntl, os, pty, re, select, signal, struct, termios, tempfile, sys, time

BIN = "/Users/priyanshuboruah/ai-native-js-runtime/sofuu"
SIZES = [(157, 38), (120, 30), (80, 24)]

BOX = "╭╮╰╯│"
CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def reconstruct(buf: bytes):
    """Replay the raw pty stream onto {row: {col: ch}} (1-based; sparse cols)."""
    text = buf.decode("utf-8", "replace")
    screen = {}
    row, col = 1, 1
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\x1b":
            m = CSI.match(text, i)
            if m:
                params, final = m.group(1), m.group(2)
                if final == "H":
                    p = params.split(";")
                    r = int(p[0]) if p and p[0].isdigit() else 1
                    c = int(p[1]) if len(p) > 1 and p[1].isdigit() else 1
                    row, col = max(1, r), max(1, c)
                elif final == "J" and params in ("2", "3"):
                    screen = {}
                elif final == "K":
                    cells = screen.setdefault(row, {})
                    for cc in [c for c in cells if c >= col]:
                        del cells[cc]
                i = m.end()
                continue
            m = OSC.match(text, i)
            if m:
                i = m.end()
                continue
            if i + 1 < n and text[i + 1] in "()" and i + 2 < n:
                i += 3  # charset designation ESC ( X — consume all 3 bytes
                continue
            i += 2  # other escape forms: consume ESC + 1
            continue
        if ch == "\r":
            col = 1
        elif ch == "\n":
            row += 1
        elif ch >= " ":
            screen.setdefault(row, {})[col] = ch
            col += 1
        i += 1
    return screen


def check(cols, rows_, buf):
    screen = reconstruct(buf)
    lines = {r: "".join(screen[r].get(c, " ") for c in range(1, max(screen[r]) + 1)).rstrip()
             for r in sorted(screen) if screen[r]}
    welcome = next((r for r in sorted(lines) if "Welcome to Sofuu!" in lines[r]), None)
    if welcome is None:
        return ["welcome panel title not found on screen"], []
    top = bot = welcome
    while top - 1 in lines and any(g in lines[top - 1] for g in BOX):
        top -= 1
    while bot + 1 in lines and any(g in lines[bot + 1] for g in BOX):
        bot += 1
    box_rows = [(r, lines[r]) for r in range(top, bot + 1) if r in lines]

    problems = []
    if len(box_rows) < 8:
        problems.append(f"expected the full box (>=8 box rows), saw {len(box_rows)}")
    edges = []
    for r, line in box_rows:
        idx = [c for c, ch in enumerate(line) if ch in BOX]
        edges.append((r, idx[0], idx[-1]))
    lefts = {e[1] for e in edges}
    rights = {e[2] for e in edges}
    if len(lefts) != 1:
        problems.append(f"left border column zigzags: {sorted(lefts)} (rows {[(e[0], e[1]) for e in edges]})")
    if len(rights) != 1:
        problems.append(f"right border column zigzags: {sorted(rights)} (rows {[(e[0], e[2]) for e in edges]})")
    if not problems and edges:
        # 0-based col 2 == screen col 3 (the renderer's GUTTER indent); the
        # right border belongs one spare cell inside the last column.
        if edges[0][1] != 2:
            problems.append(f"left border at screen col {edges[0][1] + 1}, expected 3")
        if edges[0][2] != cols - 2:
            problems.append(f"right border at screen col {edges[0][2] + 1}, expected {cols - 1}")
    return problems, box_rows, lines


def run(cols, rows_):
    tmpd = tempfile.mkdtemp(prefix="sofuu_panel_")
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["HOME"] = tmpd
        os.environ["SOFUU_HOME"] = tmpd
        os.environ["TERM"] = "xterm-256color"
        os.chdir(tmpd)
        os.execv(BIN, [BIN, "chat"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows_, cols, 0, 0))
    buf = b""
    deadline = time.time() + 8.0
    # Drain until the stream goes quiet — killing mid-repaint leaves a
    # half-written row (EL clears the line, text arrives in a later write).
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.8)
        if not r:
            break
        try:
            d = os.read(fd, 65536)
        except OSError:
            break
        if not d:
            break
        buf += d
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    os.close(fd)
    return buf


fail = False
for cols, rows_ in SIZES:
    buf = run(cols, rows_)
    problems, box_rows, lines = check(cols, rows_, buf)
    print(f"=== {cols}x{rows_}: {'PASS' if not problems else 'FAIL'} ({len(box_rows)} box rows) ===")
    for r, line in box_rows:
        print(f"  [{r:02d}] {line}")
    if problems:
        print("  --- full reconstructed screen ---")
        for r in sorted(lines):
            print(f"  |{lines[r]}")
        tail = buf.decode("utf-8", "replace")[-200:]
        print(f"  --- raw tail: {tail!r}")
    for p in problems:
        print(f"  !! {p}")
        fail = True
    print()

sys.exit(1 if fail else 0)
