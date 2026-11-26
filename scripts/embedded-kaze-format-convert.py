#!/usr/bin/env python3
"""Rewrite a Kaze Format 2 module as a legacy Format 1 or Format 4 module.

The C VM must read every format the product has ever written. Only Format 2 is
still produced by the Rust compatibility writer; Formats 1 and 4 are the two
other structural variants:

  * v1: no root origin, no per-item origin, no line table
  * v2: no root origin, per-item line table (the writer's current format)
  * v4: root origin plus a per-item origin, no line table

This converter preserves the executable code and constants verbatim, so a
diagnostic line difference between the outputs is a real VM difference and
not a recompilation artefact.

    embedded-kaze-format-convert.py <in-v2.kazebc> <out-prefix>
"""

import struct
import sys


def read_module(blob):
    def u16(at):
        return struct.unpack_from("<H", blob, at)[0], at + 2

    def u32(at):
        return struct.unpack_from("<I", blob, at)[0], at + 4

    def f64(at):
        return struct.unpack_from("<d", blob, at)[0], at + 8

    def text(at):
        size, at = u32(at)
        return blob[at:at + size], at + size

    if blob[:8] != b"KAZEBC1\0":
        raise SystemExit("not a Kaze module")
    version, at = u16(8)
    if version != 2:
        raise SystemExit("expected a Format 2 input, got %d" % version)
    needs = []
    count, at = u32(at)
    for _ in range(count):
        value, at = text(at)
        needs.append(value)
    consts = []
    count, at = u32(at)
    for _ in range(count):
        kind = blob[at]
        at += 1
        if kind == 0:
            value, at = f64(at)
        elif kind == 1:
            value, at = text(at)
        else:
            value, at = blob[at], at + 1
        consts.append((kind, value))
    caps = []
    count, at = u32(at)
    for _ in range(count):
        family, at = text(at)
        op, at = text(at)
        argc, at = blob[at], at + 1
        caps.append((family, op, argc))
    items = []
    count, at = u32(at)
    for _ in range(count):
        tag, at = blob[at], at + 1
        period, at = f64(at)
        line, at = u32(at)
        slots, at = blob[at], at + 1
        name, at = text(at)
        params = []
        nparams, at = u32(at)
        for _ in range(nparams):
            value, at = text(at)
            params.append(value)
        ncode, at = u32(at)
        code = struct.unpack_from("<%dI" % ncode, blob, at)
        at += 4 * ncode
        nlines, at = u32(at)
        lines = struct.unpack_from("<%dI" % nlines, blob, at)
        at += 4 * nlines
        items.append((tag, period, line, slots, name, params, code, lines))
    return needs, consts, caps, items


def text(out, value):
    if isinstance(value, str):
        value = value.encode()
    return out + struct.pack("<I", len(value)) + value


def build(version, needs, consts, caps, items):
    out = bytearray(b"KAZEBC1\0") + struct.pack("<H", version)
    if version >= 4:
        out = text(out, "<converted>")
    out += struct.pack("<I", len(needs))
    for value in needs:
        out = text(out, value)
    out += struct.pack("<I", len(consts))
    for kind, value in consts:
        out.append(kind)
        if kind == 0:
            out += struct.pack("<d", value)
        elif kind == 1:
            out = text(out, value)
        else:
            out.append(value)
    out += struct.pack("<I", len(caps))
    for family, op, argc in caps:
        out = text(out, family)
        out = text(out, op)
        out.append(argc)
    out += struct.pack("<I", len(items))
    for tag, period, line, slots, name, params, code, lines in items:
        out.append(tag)
        out += struct.pack("<dI", period, line)
        out.append(slots)
        out = text(out, name)
        if version >= 4:
            out = text(out, "<converted>")
        out += struct.pack("<I", len(params))
        for value in params:
            out = text(out, value)
        out += struct.pack("<I", len(code))
        for word in code:
            out += struct.pack("<I", word)
        if version == 2:
            out += struct.pack("<I", len(lines))
            for value in lines:
                out += struct.pack("<I", value)
    return bytes(out)


def synth_div0():
    # Format 2 fixture: main (item line 2) executes Const 1.0, Const 0.0, Div,
    # RetUnit with every pc mapped to statement line 4.
    code = [0, 0, 0, 1, 10, 25]
    lines = [4] * len(code)
    return build(
        2,
        ["policy"],
        [(0, 1.0), (0, 0.0)],
        [],
        [(2, 0.0, 2, 0, "main", [], code, lines)],
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--synth-div0":
        blob = synth_div0()
        with open(sys.argv[2], "wb") as handle:
            handle.write(blob)
        print("%s: %d bytes" % (sys.argv[2], len(blob)))
        return 0
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 64
    blob = open(sys.argv[1], "rb").read()
    needs, consts, caps, items = read_module(blob)
    for version in (1, 4):
        path = "%s-v%d.kazebc" % (sys.argv[2], version)
        with open(path, "wb") as handle:
            handle.write(build(version, needs, consts, caps, items))
        print("%s: %d bytes" % (path, len(build(version, needs, consts, caps, items))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
