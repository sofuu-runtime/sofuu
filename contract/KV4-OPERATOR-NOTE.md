# KV4 operator note — the quantized GPU KV cache (`--kv-dtype 8|4`)

**Status: opt-in, measured slower than f32 at every context tested, and it cannot slide past the window.
The default is f32 and stays f32.** Full receipt: `kv-evidence/REPORT-kvlong.md` (quiet M2 Pro, 6 runs per
arm). Closed-result entry: `benchmark/PROTOCOL.md` → "Closed: quantized KV on the GPU decode path
(2026-10-28)".

## What the flag does, and when it arms

`--kv-dtype 8|4` (env `SOFUU_KV_DTYPE=8|4`) arms kv4 on the GPU decode path: attended history is packed
into 32-token KIVI-layout blocks, and only the most recent 192 rows stay f32 (the residual ring the existing
f32 buffers are re-pointed at). It arms only when all of these hold — `final_seq_len > 192`, the GPU client
is allowed and built, the head geometry is kv4-compatible, and the model is resident. Otherwise the run stays
on f32 and says so:

```
[kv4] 8 KV cache requested, but it does not apply here (gpu_allowed=... geometry_ok=... ctx=... resident=...) - keeping the f32 KV cache
[kv4] 8 KV cache: 32 -token blocks, f32 residual ring 192 rows, logical window 4096 (--kv-dtype f32 restores the byte-identical f32 cache)
```

With the cache armed the CLI prints one receipt per run; use it to prove engagement in your own logs (it
appeared 6/6 runs of every quantized arm in the measurement):

```
[kv4] quantized KV armed: 8-bit, 128 blocks x 32 tokens, f32 ring 192 rows, blocks 280MB vs 896MB for the f32 cache (3.2x), 2560 B/token/layer
```

4-bit reads `168MB ... (5.3x), 1536 B/token/layer`.

## The verdict, with numbers

Qwen3-0.6B Q4_K_M, default 4096 window, GPU prefill + MMA8 GEMM flags, 32 decode tokens, serial decode,
prompts 493 / 1497 / 2993 / 3994 tokens, 3 rounds of ABBA = 6 runs per arm. Decode rate is the median of the
per-token `forward_ns` from `--trace`.

| prompt tok | f32 decode tk/s | 8-bit | 4-bit | 8-bit delta | 4-bit delta | f32 prefill tk/s | 8-bit | 4-bit |
|---|---|---|---|---|---|---|---|---|
| 493 | 111.6 | 100.1 | 98.3 | -10.3% | -12.0% | 1616 | 603 | 605 |
| 1497 | 86.3 | 69.6 | 69.8 | -19.3% | -19.1% | 1412 | 278 | 275 |
| 2993 | 66.2 | 46.9 | 47.5 | -29.2% | -28.3% | 1058 | 146 | 148 |
| 3994 | 56.3 | 38.8 | 38.6 | -31.0% | -31.4% | 874 | 111 | 111 |

Prefill deltas: -62.7% / -80.3% / -86.2% / -87.3% (8-bit) and -62.6% / -80.5% / -86.0% / -87.3% (4-bit).
Absolute prefill time at ~4k tokens goes 0.30 s → 35.9 s.

## Why the gap grows: it is a slope, not an offset

Decode ms/token fits `a + b*ctx` with r2 >= 0.999 in all three arms:

| arm | a (fixed ms/token) | b (us per context token) | r2 |
|---|---|---|---|
| f32 | 7.76 | 2.49 | 0.99938 |
| 8-bit | 7.69 | 4.53 (**1.82x**) | 0.99984 |
| 4-bit | 7.78 | 4.49 (**1.80x**) | 0.99914 |

The fixed per-token cost is identical, so the arms differ only in attention. The quantized attention kernel
has 1.82x f32's slope while reading 3.2x (8-bit) / 5.3x (4-bit) fewer bytes. It is therefore **not
bandwidth-bound**: 4-bit and 8-bit land within ±2% of each other in time even though 4-bit reads 1.7x fewer
bytes. Dequant ALU/occupancy cost dominates the byte saving, so there is no context length at which this
kernel turns into a win — as ctx → ∞ the penalty approaches 1.82x ms/token, about -45% tk/s. It is already
-31% at 4096. The f32 arm at these lengths uses `llama_attn_w32`, the same kernel family and threadgroup
geometry as `llama_attn_w32_kv4`, so this is like-for-like.

The same shape, smaller, on Llama-3.2-1B Q8_0 (64 KiB KV/token vs Qwen's 224): -13.4% decode at 1497 and
-20.9% at 2993, prefill -53.3% / -65.9%, slope 2.07x. There is a byte-count term — the model with the smaller
KV loses less — but the sign never flips.

## Hard stop at the window: packed blocks cannot slide

History is addressed by absolute position, so compaction (which drops middle tokens and shifts survivors
down) cannot be expressed. Crossing 4095 with kv4 armed **aborts the process** (SIGABRT, rc = -6) after ~100
tokens of already-streamed output, for both 8-bit and 4-bit:

```
[kv4] FATAL: decode reached the 4096-token logical window with the quantized KV cache armed; the packed blocks cannot slide.
[kv4] Packed blocks are indexed by absolute position and cannot slide: a sliding history needs the f32 cache - rerun with --kv-dtype f32. Raising --ctx does not help: at 8192 the GPU decode client does not arm on a 16 GB M2 Pro, so the run quietly decodes on the CPU.
```

The f32 arm on the identical input finishes all 200 tokens normally (rc = 0, 52.7 tk/s) because it can slide.
A prompt longer than the window is refused cleanly by both arms (rc = 1), which is not kv4-specific.

## The `--ctx` trap

"`--ctx 8192` will give me room" is false on this class of machine. At 8192 the GPU decode client fails to
build even for f32 — `decode client NOT armed: building it failed ... this model decodes on the CPU` — and the
kv4 arm additionally prints `the quantized cache could not be armed - restoring the full f32 KV cache (8192
tokens)`. A user who follows that advice gets a CPU-path run (an order of magnitude slower), not a longer
quantized window. kv4 above 4096 is unreachable here; do not document raising `--ctx` as the way out.

## Memory: the one defensible claim, and the retracted one

- The KV region really is smaller: 896 MB f32 vs 280 MB (8-bit) vs 168 MB (4-bit) at ctx 4096 — 3.2x / 5.3x,
  from the binary's own receipt (2560 / 1536 B per token per layer × 28 layers).
- Peak **phys_footprint** falls with it: 1830 MB f32 → 1701 MB (8-bit) at a 493-token prompt, 2134 → 1704 MB
  at 3994 (per-arm spread 1-3 MB). "8-bit saves ~130 MB at ~500 tokens and ~430 MB at ~4k" is supportable.
- Peak **RSS** is not: at long prompts the *same arm* reads 1798 MB or 1283 MB depending on the rep, and
  4-bit/8-bit read identically. Report footprint deltas, never RSS deltas at long context, and do not claim
  the full ~570 MB the buffer arithmetic predicts (footprint recovers part of it, and the extra ~112 MB 4-bit
  should save over 8-bit measures as 0 MB).

## Operating rules

- Keep the default f32 unless you are measuring memory and can pay the time. Never promote 8-bit or 4-bit on
  throughput.
- Stay inside the window: a kv4 run that would cross it aborts, and mid-generation that means a truncated
  answer plus a crash signal.
- Do not answer "`--ctx`" for a sliding requirement; the answer is `--kv-dtype f32`.
- Verify engagement before trusting any kv4 number: the armed receipt must print, and logprobs must differ
  from a f32 run (they differ on 30-31 of 31 steps; 8-bit reproduces f32's greedy tokens 31/31, 4-bit does not
  — it diverged 8/31 at 493 tokens).
- The loss is a property of this kernel's dequant-by-threadgroup structure, not of quantization in principle.
  Re-attempt only with a kernel whose *slope* matches f32; fewer bytes alone is not the lever.
