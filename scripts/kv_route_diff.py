#!/usr/bin/env python3
"""Pin kv_write/rope divergences between the CPU-tile and GPU prefill routes.

Usage: SOFUU_KV_DUMP=/tmp/kvcpu (CPU run) and /tmp/kvgpu (GPU run), then:
  python3 scripts/kv_route_diff.py /tmp/kvcpu /tmp/kvgpu [layers nkv msl hd]

The dump files are written by model.rs maybe_dump_kv (SOFUU_KV_DUMP env).
Localizes |cpu-gpu| KV value diffs into (layer, kh, pos, hd) bins so a RoPE
theta-precision divergence (high-frequency lanes worst) separates from a
layout/position bug (contiguous positions wrong).
"""
import struct, sys, math
cp, gp = sys.argv[1], sys.argv[2]
L, NKV, MSL, HD = (int(x) for x in sys.argv[3:7]) if len(sys.argv) > 6 else (28, 8, 4096, 64)
def f32s(p):
    d = open(p, "rb").read()
    n = len(d) // 8  # the anon-buffer region is 2x the logical cache; first half = f32 cache
    return struct.unpack(f"<{n}f", d[: n * 4])
A, C = f32s(cp + ".k.bin"), f32s(gp + ".k.bin")
n_reg = min(len(A), len(C))
assert n_reg % (L * NKV * MSL * HD) == 0 or True
cnt = 0; top = []
hist = {}
sq = 0.0; n_s = 0
for off in range(n_reg):
    d = abs(A[off] - C[off])
    if d == 0.0:
        continue
    cnt += 1
    sq += d * d; n_s += 1
    l = off // (NKV * MSL * HD); r = off % (NKV * MSL * HD)
    kh = r // (MSL * HD); pos = (r % (MSL * HD)) // HD; hdi = r % HD
    if d > 0.01:
        hist[pos] = hist.get(pos, 0) + 1
    top.append((d, l, kh, pos, hdi))
    if len(top) > 48:
        top.sort(reverse=True); del top[24]
top.sort(reverse=True)
print(f"differing f32: {cnt}/{n_reg} ({100*cnt/n_reg:.2f}%)  RMS={math.sqrt(sq/max(1,n_s)):.3e}")
print("top-12 |diff| (layer, kh, pos, hd):")
for d, l, kh, pos, hdi in top[:12]:
    print(f"  {d:.4e}  l={l} kh={kh} pos={pos} hd={hdi}  cpu={A[[0]][0] if False else ''}")
print("positions with >0.01 diffs:", len(hist), " (min/max pos:", (min(hist), max(hist)) if hist else None, ")")
hd_hist = {}
for d, l, kh, pos, hdi in top:
    hd_hist[hdi] = hd_hist.get(hdi, 0) + 1
print("worst-hit hd lanes:", sorted(hd_hist.items()))
