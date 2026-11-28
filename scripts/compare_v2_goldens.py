#!/usr/bin/env python3
"""Compare a v2 emulator capture against a v1 golden.

The v1 goldens include Android's own status-bar icons and the emulator paints
its own chrome, so a full-frame pixel gate would be dishonest. This tool
reports two numbers instead:

* ``full``  — every pixel, including the system chrome bands.
* ``app``   — the app area only, with ``--ignore-top`` / ``--ignore-bottom``
  pixels removed (default 63 px = 24 dp at 2.625x).

It also prints the bounding box of pixels whose largest RGB delta exceeds
``--threshold`` so remaining atlas/spacing deltas are visible, not hidden.

Usage:
  scripts/compare_v2_goldens.py v1.png v2.png [--ignore-top 63] [--ignore-bottom 63]
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib


def read_png(path: str) -> tuple[int, int, int, bytes]:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos = 8
    idat = b""
    width = height = 0
    bit_depth = color_type = 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if interlace:
                raise ValueError(f"{path}: interlaced PNGs are not supported")
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    if bit_depth != 8:
        raise ValueError(f"{path}: only 8-bit PNGs are supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"{path}: unsupported color type {color_type}")
    raw = zlib.decompress(idat)
    stride = width * channels
    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _y in range(height):
        filt = raw[p]
        p += 1
        line = bytearray(raw[p : p + stride])
        p += stride
        if filt == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif filt == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up = prev[i]
                up_left = prev[i - channels] if i >= channels else 0
                estimate = left + up - up_left
                pa = abs(estimate - left)
                pb = abs(estimate - up)
                pc = abs(estimate - up_left)
                predictor = left if (pa <= pb and pa <= pc) else (up if pb <= pc else up_left)
                line[i] = (line[i] + predictor) & 255
        elif filt != 0:
            raise ValueError(f"{path}: bad filter {filt}")
        out += line
        prev = line
    return width, height, channels, bytes(out)


def compare(a_path: str, b_path: str, ignore_top: int, ignore_bottom: int, threshold: int) -> int:
    aw, ah, ac, ap = read_png(a_path)
    bw, bh, bc, bp = read_png(b_path)
    if (aw, ah) != (bw, bh):
        print(f"size mismatch: {a_path} {aw}x{ah} vs {b_path} {bw}x{bh}")
        return 1
    if ac < 3 or bc < 3:
        print("need RGB or RGBA inputs")
        return 1

    def rgb(buf: bytes, channels: int, x: int, y: int) -> tuple[int, int, int]:
        i = (y * aw + x) * channels
        return buf[i], buf[i + 1], buf[i + 2]

    def stats(y0: int, y1: int) -> tuple[float, int, tuple[int, int, int, int] | None]:
        total = 0
        over = 0
        min_x = aw
        min_y = ah
        max_x = -1
        max_y = -1
        for y in range(y0, y1):
            for x in range(aw):
                ar, ag, ab = rgb(ap, ac, x, y)
                br, bg, bb = rgb(bp, bc, x, y)
                d = max(abs(ar - br), abs(ag - bg), abs(ab - bb))
                total += abs(ar - br) + abs(ag - bg) + abs(ab - bb)
                if d > threshold:
                    over += 1
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        pixels = max(1, (y1 - y0) * aw)
        mae = total / (pixels * 3.0)
        bbox = None if max_x < 0 else (min_x, min_y, max_x, max_y)
        return mae, over, bbox

    full_mae, full_over, full_bbox = stats(0, ah)
    app_y0 = min(max(ignore_top, 0), ah)
    app_y1 = max(min(ah - ignore_bottom, ah), app_y0)
    app_mae, app_over, app_bbox = stats(app_y0, app_y1)

    print(f"golden: {a_path}")
    print(f"capture: {b_path}")
    print(f"size: {aw}x{ah}")
    print(f"full: mae={full_mae:.2f} over_threshold={full_over} bbox={full_bbox}")
    print(
        f"app:  mae={app_mae:.2f} over_threshold={app_over} bbox={app_bbox} "
        f"(rows {app_y0}..{app_y1})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden")
    parser.add_argument("capture")
    parser.add_argument("--ignore-top", type=int, default=63)
    parser.add_argument("--ignore-bottom", type=int, default=63)
    parser.add_argument("--threshold", type=int, default=24)
    args = parser.parse_args()
    return compare(
        args.golden, args.capture, args.ignore_top, args.ignore_bottom, args.threshold
    )


if __name__ == "__main__":
    sys.exit(main())
