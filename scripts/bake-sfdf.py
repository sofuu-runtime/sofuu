#!/usr/bin/env python3
"""Bake TrueType faces into compact SDF atlases for `sofuu-ui`.

The runtime never rasterizes text on the CPU: `crates/sofuu-ui/src/text.rs` reads
these atlases, and the GPU/software rasterizers sample the single-channel
distance field analytically at any DPI. This script is the offline half.

Formats
-------
v1 (`SFDF`) - one advance per font (monospace). The original Geist Mono bake.

v2 (`SFD2`) - per-glyph advance + cell placement, so proportional faces keep
their real side bearings.

v3 (`SFD3`) - a v2 header plus a block table, so ONE R8 texture carries every
family/weight. Header:

    header   32 B   magic, width, height, px_range, total_glyphs,
                    fallback_advance_em, ascender_em, line_height_em
    blocks   12 B   preceded by block_count u32; per block: family u8
                    (0 sans, 1 mono), pad u8, weight u16, first_entry u32,
                    entry_count u32
    glyph    36 B   uv[4], offset_x_em, offset_y_em, quad_w_em, quad_h_em,
                    advance_em
    pixels   W*H B  row-major R8 (128 = contour edge, >128 = inside)

Block order is declared by the caller:

    --face sans:400:Geist-Regular.ttf
    --face sans:500:Geist-Medium.ttf
    --face sans:600:Geist-SemiBold.ttf
    --face sans:700:Geist-Bold.ttf
    --face mono:400:GeistMono-Regular.ttf

`text.rs` maps `(family, weight)` onto those blocks and falls back to block 0
when a single-block atlas is supplied.

Usage
-----
    python3 scripts/bake-sfdf.py --version 3 \
        --face sans:400:path/Geist-Regular.ttf \
        --face mono:400:path/GeistMono-Regular.ttf \
        --out crates/sofuu-ui/assets/ui_fonts.sfdf
"""

from __future__ import annotations

import argparse
import math
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FIRST = 0x20
LAST = 0x7E
GLYPHS = LAST - FIRST + 1  # 95 printable ASCII
FALLBACK_INDEX = GLYPHS  # 96th entry: .notdef
COUNT = GLYPHS + 1


def render_mask(font: ImageFont.FreeTypeFont, ch: str, cell_w: int, cell_h: int,
                baseline: int, pad_x: int) -> np.ndarray:
    """Rasterize one glyph into a binary mask at the mask resolution."""
    img = Image.new("L", (cell_w, cell_h), 0)
    draw = ImageDraw.Draw(img)
    # anchor "ls" = left/baseline, so the baked cell needs no per-glyph fudge.
    draw.text((pad_x, baseline), ch, font=font, fill=255, anchor="ls")
    return np.asarray(img, dtype=np.uint8) >= 128


def sdf_from_mask(mask: np.ndarray, px_range: float) -> np.ndarray:
    """Signed distance (positive inside) encoded as R8 with 128 on the contour."""
    inside = mask.astype(bool)
    outside = ~inside
    d_out = ndimage.distance_transform_edt(outside)
    d_in = ndimage.distance_transform_edt(inside)
    signed = d_in - d_out  # in mask pixels
    v = 128.0 + 127.0 * np.clip(signed / px_range, -1.0, 1.0)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def bake_face(font_path: str, em_px: int, px_range: float, hi: int = 4):
    """Bake one face. Layout math runs in supersampled space, then the field is
    downsampled to one distance value per cell pixel."""
    font = ImageFont.truetype(font_path, em_px * hi)
    ascent_h, descent_h = font.getmetrics()
    pad_x_h = max(4 * hi, round(em_px * hi * 0.34))
    pad_y_h = max(4 * hi, round(em_px * hi * 0.34))
    advances_h = [font.getlength(chr(c)) for c in range(FIRST, LAST + 1)]
    cell_w_h = int(math.ceil(pad_x_h * 2 + max(advances_h) + 1))
    cell_h_h = int(math.ceil(pad_y_h * 2 + ascent_h + descent_h))
    baseline_h = pad_y_h + ascent_h

    cell_w = int(math.ceil(cell_w_h / hi))
    cell_h = int(math.ceil(cell_h_h / hi))
    cells = []
    for idx in range(COUNT):
        ch = chr(FIRST + idx) if idx < GLYPHS else "?"
        # A little extra room so italic/overhanging ink cannot clip.
        mask = render_mask(
            font, ch, cell_w_h + hi, cell_h_h + hi, baseline_h, pad_x_h
        )
        sdf = sdf_from_mask(mask, px_range * hi)
        sdf = sdf[hi // 2:hi // 2 + cell_h * hi:hi, hi // 2:hi // 2 + cell_w * hi:hi]
        assert sdf.shape == (cell_h, cell_w), (ch, sdf.shape, cell_h, cell_w)
        cells.append(sdf)
    return {
        "cell_w": cell_w,
        "cell_h": cell_h,
        "pad_x": pad_x_h / hi,
        "baseline": baseline_h / hi,
        "ascent": ascent_h / hi,
        "descent": descent_h / hi,
        "advances": [a / hi for a in advances_h] + [font.getlength("?") / hi],
        "cells": cells,
        "px_range": px_range,
    }


def pack(faces, cols: int):
    rows_per_face = math.ceil(COUNT / cols)
    cell_w = max(f["cell_w"] for f in faces)
    cell_h = max(f["cell_h"] for f in faces)
    width = cols * cell_w
    height = rows_per_face * len(faces) * cell_h
    atlas = np.zeros((height, width), dtype=np.uint8)
    entries = []  # (face_idx, glyph_idx, uv) -> flat list
    for fi, f in enumerate(faces):
        for gi, cell in enumerate(f["cells"]):
            col = gi % cols
            row = fi * rows_per_face + gi // cols
            x0, y0 = col * cell_w, row * cell_h
            # Center smaller cells inside the shared cell box so metrics stay
            # consistent for every face of the family.
            dx = (cell_w - f["cell_w"]) // 2
            dy = (cell_h - f["cell_h"]) // 2
            atlas[y0 + dy:y0 + dy + f["cell_h"], x0 + dx:x0 + dx + f["cell_w"]] = cell
            uv = [x0 / width, y0 / height, (x0 + cell_w) / width, (y0 + cell_h) / height]
            entries.append((fi, gi, uv))
    return atlas, entries, width, height, cell_w, cell_h


def write_atlas(path: str, version: int, faces, cols: int, line_height_em: float):
    atlas, entries, width, height, cell_w, cell_h = pack(faces, cols)
    px_range = faces[0]["px_range"]
    em = float(faces[0]["em_px"])
    fallback_advance = max(f["advances"][-1] for f in faces) / em
    ascender = max(f["ascent"] for f in faces) / em

    magic = {1: b"SFDF", 2: b"SFD2", 3: b"SFD3"}[version]
    out = bytearray(magic)
    out += struct.pack(
        "<IIfIfff",
        width,
        height,
        px_range,
        len(entries),
        fallback_advance,
        ascender,
        line_height_em,
    )
    if version == 3:
        out += struct.pack("<I", len(faces))
        for f in faces:
            family_tag = 1 if f["family"] == "mono" else 0
            out += struct.pack("<BBHII", family_tag, 0, int(f["weight"]), 0, COUNT)
        # Patch the per-block first_entry now that all headers are written.
        block_bytes = 12
        base = 36
        for fi in range(len(faces)):
            struct.pack_into("<I", out, base + fi * block_bytes + 4, fi * COUNT)
    entry_bytes = 32 if version == 1 else 36
    for fi, gi, uv in entries:
        f = faces[fi]
        off_x = -((cell_w - f["cell_w"]) / 2 + f["pad_x"]) / em
        off_y = -((cell_h - f["cell_h"]) / 2 + f["baseline"]) / em
        quad_w = cell_w / em
        quad_h = cell_h / em
        out += struct.pack("<4f4f", *uv, off_x, off_y, quad_w, quad_h)
        if version >= 2:
            out += struct.pack("<f", f["advances"][gi] / em)
    out += atlas.tobytes()
    with open(path, "wb") as fh:
        fh.write(out)
    print(
        f"baked {path}: {width}x{height} ({len(out)} B), "
        f"{len(entries)} glyphs in {len(faces)} face(s), cell {cell_w}x{cell_h}, "
        f"px_range {px_range}"
    )


def parse_face(spec: str):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected family:weight:path")
    family, weight, path = parts
    return family, int(weight), path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", action="append", type=parse_face, required=True,
                    help="family:weight:path (e.g. sans:400:Geist-Regular.ttf)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--em", type=int, default=26)
    ap.add_argument("--px-range", type=float, default=3.5)
    ap.add_argument("--cols", type=int, default=32)
    ap.add_argument("--line-height-em", type=float, default=1.3)
    args = ap.parse_args()

    faces = []
    for family, weight, path in args.face:
        f = bake_face(path, args.em, args.px_range)
        f["em_px"] = args.em
        f["family"] = family
        f["weight"] = weight
        faces.append(f)
    write_atlas(args.out, args.version, faces, args.cols, args.line_height_em)


if __name__ == "__main__":
    main()
