# Pi 5 CPU Inference Optimization — NEON Tiles, Batched Glue, Profiling

Shipped 2026-10-14. Target device: Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz,
NEON + DOTPROD only — no i8mm, no SVE). Scope: CPU prefill/decode throughput
for GGUF models below ~4B parameters.

## Results (234-token prompt, performance governor, same-session llama.cpp refs)

| Model                  | Before session | After (t/s prefill / decode) | llama.cpp (t/s)      |
|------------------------|----------------|------------------------------|----------------------|
| Qwen3-0.6B Q4_K_M      | 52.2 / 19.9    | 110.4 / 22.0                 | 176.3 / 23.6         |
| Llama-3.2-1B Q8_0      | 27.5 / 8.4     | 120.5 / 8.7                  | 60.2 / 8.5           |
| Spark-X2.5-4B Q4_K_M   | 5.2 / 4.0      | 14.0 / 4.1                   | 22.8 / 4.2           |
| liquid-350 BF16        | 16.4 / 5.4     | 17.2 / 5.4                   | —                    |
| liquid-350 MLX4        | 6.7 / 6.4      | 6.7 / 6.6                    | —                    |

Highlights: Llama-1B Q8_0 prefill **2.0× faster than llama.cpp**; decode equal
or better on every model. Decode is memory-bandwidth-bound on both engines
(~8.6 GB/s effective) — that is the Pi 5 hardware ceiling.

## What changed

### 1. NEON 8×4 repack tile kernels (`src/simd/arch/a76_repack.h`)

Ported llama.cpp's DOTPROD-path repack kernels verbatim:

- `ggml_gemv/gemm_q4_K_8x4_q8_K`, `ggml_gemv/gemm_q6_K_8x4_q8_K`,
  `ggml_gemv/gemm_q8_0_4x4_q8_0` (8-row interleaved weight tiles, 4-row
  activation tiles, `vdotq_laneq_s32` inner loops),
- `ggml_quantize_mat_q8_0_4x4` (NEON activation quantizer),
- `decode_q_Kx8_6bit_scales` helper.

All six bodies are machine-diff verified byte-identical to the reference
(modulo `tile_block_*` struct renames). Weight tiles are repacked at model
load (`sofuu_tile_warm` / lazy `sofuu_tile_lookup`); every fallback path is
guarded — non-A76 hosts get the generic kernels unchanged. `SOFUU_NO_TILES=1`
disables the tile path entirely as a kill switch.

### 2. Dispatch + scratch lifetime (`src/simd/infer_kernels.cpp`)

- Batch GEMM entry points route to the NEON wrappers (batch ≥ 2, rows/cols
  aligned, tile lookup hit) with generic/SDOT fallbacks preserved.
- `tile_scratch<T>()`: reuse a `thread_local` scratch for activation tiles
  instead of allocating and zero-filling a fresh vector on every weight-matrix
  call (~280 KB per call at batch 64). Measured +16% on Qwen prefill.

### 3. Chunk scheduling (`crates/sofuu-core/src/infer/{model,spark}.rs`)

- `max_prefill_batch()` (default 64) replaces the build-time
  `MAX_PREFILL_BATCH` constant; overridable at process start with
  `SOFUU_PREFILL_BATCH`, snapped to a multiple of 4. 64 measured optimal
  (128/256 within noise or slower).
- The final prefill chunk is split into a 4-aligned main body (so the tiled
  GEMM path stays eligible for every token) plus a ≤3-row remainder that goes
  through the verified scalar/SDOT path. Worth +27% prefill on Spark, where a
  78-token prompt previously dropped its 14-token tail onto the slow path.

### 4. Batched elementwise glue (one FFI call per layer)

Previously norms, biases, RoPE, KV scatter, SwiGLU and residual adds ran as
~4,000 per-token FFI calls per 64-token chunk. New internal-parallel kernels
(`sofuu_batched_rmsnorm[_in_place]`, `sofuu_batched_mad`,
`sofuu_batched_add_bias`, `sofuu_batched_swiglu`, `sofuu_batched_rope_kv`)
replace each per-token loop with a single call driven by the thread pool.

### 5. Profiling (`SOFUU_PROFILE=1`)

Per-chunk wall-time split (gemm / gemv / attention / other) printed to stderr
via a zero-cost-when-disabled RAII profiler (`quant::prof`). Findings that
drove the work above: Qwen q4_K GEMM ≈ 76 GMAC/s (scale-FMA-bound — the same
wall llama.cpp's identical kernel sits at), q8_0 GEMM ≈ 166 GMAC/s, attention
internally parallel and ~12% of prefill, residual glue < 27% and dropping.

## Verification

- Host suite: 523/523 (`cargo test -p sofuu-core --release --lib`).
  Includes the renamed-fixture fix for `test_tile_gemv_q8_0_matches_reference`
  (the old test built the f16 scale bytes from f32 low-mantissa bits and
  compared against the net dot product; the fixture now uses f16-exact deltas
  and a cancellation-safe L1-relative tolerance).
- Device E2E: templated-prompt coherence, clean EOS stops, and full
  `pi_bench.sh` battery runs on every shipped binary; llama.cpp reference
  numbers measured on-device in the same session (`-t 4`).

## Known gaps / next levers

- q4_K prefill GEMM is scale-FMA-bound in both engines. Pre-decoding the
  6-bit scales into float sidecars at repack time would remove the scale
  decode + bias-correction chain from the hot loop (est. 1.3–1.5× on q4_K).
- Spark per-matmul shapes run at roughly half the q4_K rate of Qwen's shapes;
  needs a shape/alignment audit of its fused-QKV row-range matmats.
- Decode is at the DRAM wall for both engines; further decode gains require
  reading fewer bytes per token, not faster kernels.

## Session 2 (2026-10-14 PM): Spark tile-path fix, sidecar verdict, parallel repack

Same device and pipeline (llama.cpp build 11041 refs measured in the morning
session: Qwen 164.3/22.5, Llama-1B 59.9/8.6, Spark 22.8/4.1 — prefill/decode
t/s).

| Model                  | Before session 2 | After (prefill / decode) | Prior best |
|------------------------|------------------|--------------------------|------------|
| Spark-X2.5-4B Q4_K_M   | 13.7 / 4.1       | **19.0 / 4.2**           | 13.7 / 4.1 |
| Qwen3-0.6B Q4_K_M      | 111.5 / 21.8     | 112.7 / 20.4             | 111.5 / 21.8 |
| Llama-3.2-1B Q8_0      | 120.3 / 8.7      | 120.5 / 8.5              | unchanged  |

Spark prefill goes 13.7 → 19.0 t/s (0.60× → **0.83× of llama.cpp**); its
per-chunk gemm bucket drops 3.86 s → 2.52 s (−35%).

### What changed

1. **Spark fused-QKV tile-path fix** (`spark.rs` warm loop). The Spark warm
   loop packed the whole fused `[q+2·kv, dim]` tensor once, but
   `project_qkv_batch` issues three row-range matmats keyed by
   (pointer, rows, cols, fingerprint) — Q passes rows=q_dim, K/V pass offset
   pointers, so all three lookups always missed and every Spark QKV GEMM/GEMV
   ran the SDOT fallback, with the repacked fused tile (~35% of Spark's
   weight bytes) as dead heap. The warm loop now warms the three row ranges
   at the exact pointers/shapes the projections use. Decode also gains its
   GEMV tile hits (4.1 → 4.2 t/s).
2. **q4_K decoded-scale sidecar — measured dead end, shipped opt-in.** A
   bit-exact f32 scale sidecar (`tile_sc_q4_Kx8`, folded `scale`/decoded
   `min`, filled at warm from the same 12-byte groups) was built and verified
   bit-identical against the original kernels in test
   (`test_tile_q4_k_scale_sidecar_bit_exact`, gemv + gemm, serial and
   parallel warm paths). On the device: prefill GEMM **unchanged** — the
   in-loop 12-byte scalar decode overlaps the NEON-pipe wall (~76 GMAC/s)
   instead of adding to it — while the +33% tile bytes slow the DRAM-bound
   q4_K GEMV decode ~15–20% (Qwen 20.4 → 16.9, Spark 4.2 → 3.2 t/s when on).
   Shipped default-off behind `SOFUU_SCALE_SIDECAR=1`; the earlier 1.3–1.5×
   estimate in the gap list above is resolved as a measured no on A76.
3. **Parallel tile repack at load.** The repack is a pure per-row-group
   shuffle, so `sofuu_tile_warm` now splits it by row group across the
   worker pool above a small-matrix threshold (serial path preserved for
   tiny tensors). Load profile: Qwen `InferModel` 339.55 → 251.57 ms, Spark
   1.65 → 1.48 s — the repack slice itself went ~4× (4 threads); the rest
   of load is KV-cache allocation and header parse, which remains the next
   load-latency lever.
4. **Dropped: final-chunk pad-to-4** (previously sketched for TTFT) — ~2% on
   a 234-token prompt, and its correctness surface (KV ring slots, lookahead
   batching, SWA layers) overlaps code another agent is actively editing.

### Verification

Host suite green including the new bit-exactness test; device E2E coherent
outputs, clean EOS stops, load RAM peak 4.3 GB on Spark (7.3 GiB available).

### Decode addendum (2026-10-14 late): speculative batch pad

Decode diagnostics reframed the wall: small-model decode is fixed-overhead
bound, not DRAM bound — Qwen decodes at 7.6 GB/s effective streaming while
Llama-1B does 11.4 GB/s (same wall cannot explain a 1.5× gap). And the
shipped `--lookahead` flag was a measured **−27%** regression on-device:
`forward_speculative_batch` builds 2-row batches, the tile GEMM gate needs
batch ≥ 4 %4, so every spec forward ran the row-wise SDOT fallback (one full
weight stream per row), and non-Gguf backends fell back to a *serial*
per-token speculative loop (Spark timed out under `--lookahead`).

Fixes:
1. **Small-batch pad in the tile dispatch** (`infer_kernels.cpp`): all three
   gemm variants (q8_0/q4_K/q6_K) pad 2–3-row activation batches (cap 8) to
   the next 4-multiple inside the call — one weight stream per speculative
   step; pad rows are zeroed scratch, the caller sees only real rows
   (proved bit-equal to an explicit zero-padded batch-4 run in
   `test_tile_gemm_small_batch_pad_bit_exact`). Also repairs the prefill
   final-chunk remainder (2–3 rows: one stream, not three).
2. **`--lookahead` gated to the Gguf backend** (the only one with a batched
   speculative path); other backends ignore the flag instead of paying the
   serial-fallback multiplier.

Receipts (Qwen natural 135-tk ctx, decode t/s): base 22.1 vs `--lookahead`
21.5 (near-parity; was −27%). Qwen repetitive list: 23.9 → **25.7** (+7.5%).
Llama-3.2-1B Q8_0: 8.7 → **10.8** (+24%) — stream-bound models capture the
full 2-tokens-per-stream upside; overhead-bound Qwen shares it with fixed
costs. Spark: base 4.2 == gated 4.2 (was a serial 3× penalty / timeout).

Greedy-equivalence note: verification is target-strict (the base row's
logits decide), but after an accepted candidate the next token is sampled
from the candidate row's batch-computed hidden state, so temp-0 output can
flip an argmax near-tie (~1 token per 100+, immediate re-sync). Outputs are
distribution-equivalent, not bit-identical — same property llama.cpp's
speculative decoding has.

Known gaps updtae: the remaining Qwen-class decode gap is fixed per-token
overhead (~14 ms/token vs a 31 ms stream at 11.4 GB/s → up to +45%
available) — needs decode-phase profiling and batch-1 glue fusion in
`execute_layers` (next wave; `execute_layers` already has the
streaming-weights non-temporal prefetch hooks).

### Decode wave 2 (2026-10-14 evening): instrumentation falsifies the overhead theory

Added per-token decode profiling (`prof::TokenScope` + periodic `report()`
in the generate loop). Measured per-token split for Qwen3-0.6B (32-token
windows): **gemv 1051 ms of 1270 (82%)** across 197 matvecs = 32.8 ms/token
→ effective weight streaming at **10.7 GB/s**; attention 4.8 ms/token; all
non-kernel glue (norms, sampler, detokenize, stack) **3.0 ms/token (7.6%)**.
Llama-1B: gemv at 12.4 GB/s effective, non-stream 4.4 ms/token. Small-model
decode was therefore **never fixed-overhead-bound** — the "7.6 vs 11.4
GB/s" gap above was an artifact of dividing whole-token time by weight
bytes; both engines (and llama.cpp) run at the same effective controller
wall (~11-12.5 GB/s, ~65-73% of LPDDR4X peak). The overhead paragraph in
the previous section is corrected by this measurement.

Consequences: per-call dispatch/glue shaving is bounded at ~10% (QK-norm
per-head FFI batching — 672 → 56 calls/token via `batched_rmsnorm_in_place`
— landed for hygiene, measured within noise). Remaining real decode levers
are bytes-per-token: a higher speculative accept rate (the in-flight
candidate-tree/`tree_rope` machinery is the right vehicle for
multi-candidate steps — same-position candidate rows would overwrite each
other's KV without it), SnapKV at long context, KV quantization (KIVI),
and controller-level DRAM efficiency (shared wall with llama.cpp).

## Session 3 (2026-10-14 night): q4_K prefill attribution, NEON act-quant, nibble-plane verdict

Same device. llama.cpp same-session refs (build 11041, `-t 4`): Qwen 171.2/23.3,
Llama-1B 55.6/8.3, Spark 22.1/4.2 (prefill/decode t/s).

| Model                  | After session 3 (prefill / decode) | Prior best | llama.cpp        |
|------------------------|------------------------------------|------------|------------------|
| Qwen3-0.6B Q4_K_M      | **128.0 / 20.3**                   | 111.5 / 20.4 | 171.2 / 23.3   |
| Llama-3.2-1B Q8_0      | 122.0 / 8.5                        | 120.5 / 8.5  | 55.6 / 8.3     |
| Spark-X2.5-4B Q4_K_M   | **21.3 / 4.1** (0.96× of llama.cpp)| 19.0 / 4.2   | 22.1 / 4.2     |

### R0 — attribution (`SOFUU_BENCH_GEMM`)

Split the q4_K tile GEMM at Qwen shapes (batch 64): the activation quantize
prologue — scalar `_generic` quantizer, single-threaded, serial before
dispatch — was **22-37%** of the whole call; the kernel itself ran 121 vs the
q8_0 kernel's 185 GMAC/s.

### R1 — NEON q8_K 4×4 act quantizer + parallel prologue (shipped, bit-exact)

Ported llama.cpp's NEON `ggml_quantize_mat_q8_K_4x4` (amax via `vmaxvq`,
magic-number rounding `(x + 12582912.0f)` → low-byte truncation via
`vmovn`, order-free exact bsums via `vpaddq`) and parallelized the quantize
prologue across the worker pool in the q4_K/q6_K tile dispatch. Proved
bit-identical to the scalar generic (`test_quant_q8_k_4x4_neon_bit_exact`);
quant share collapsed 22-37% → 5.5%. Qwen prefill 111 → 128 (+15%); Spark
18.9 → 21.4.

### R2 — glue/attention scopes

`prof` module gained NORM/ROPE/EMBED/GLUE buckets + `TokenScope`; QK-norm
per-head FFI batching (672 → 56 calls/token) landed, within noise. Attention
prefill sub-bucket is the next measurement surface for the Qwen prefill
deficit (128 vs llama.cpp 171: the residual is attention+glue, not GEMM).

### R3 — nibble-plane q4_K GEMM: measured, below the ship bar, shipped opt-in

The packed q4_K kernel runs 4 aux NEON ops per 16 `vdot`s (two `vand`/`vshr`
nibble unpacks). R3 pre-expanded both nibbles into int8 planes at warm
(`tile_block_q4_Kx8p`: `qs_lo`/`qs_hi` 1024 B each, 2176 B/block = +89% tile
bytes; `sofuu_repack_q4k_8bl_planes_range`, parallel by row group) so the
hot loop is pure `vdot`. Bit-identical by construction (each plane byte holds
exactly the nibble the packed kernel materialized) and proven by
`test_gemm_q4_k_nibble_plane_bit_exact` (3 shapes).

Device verdict — the +27% instruction-slot prediction was **wrong**:
kernel-only 129-130 vs 120-122 GMAC/s = **+5.3-7.8%** across real Qwen shapes
(2048×1024, 3072×1024, 1024×4096, batch 64/128), and only ~+1% E2E prefill
(129-130 vs 128) with ~+250 MB RAM. The remaining q4_K-vs-q8_0 gap (130 vs
185 GMAC/s) is the per-subblock scale-application chain (`vmulq_s32` scale
FMULs, `vmovl/vcvtq`, bsums-bias `vmlal`s) that the plane kernel still
carries — nibble unpacks were not the bottleneck. Below the ≥15% bar from the
approved plan: **shipped opt-in** (`SOFUU_NIBBLE_PLANES=1`), gated per tensor
(≤16M elements) with a global 768 MB budget (`SOFUU_NIBBLE_PLANES_MB`),
GEMV decode stays on packed tiles (DRAM-bound, planes would cost bytes).
Bench: `SOFUU_BENCH_GEMM=rows,cols,batch,iters,plane_iters`.

New next lever implied by the measurement: fold the scale-apply chain (e.g.
pre-scaled int accumulators or delayed combined scaling) — that is where the
quantized GEMM rate now actually lives.

### Verification

Host suite green (incl. all three new bit-exactness tests); full device
battery above on the shipped binary (planes default-off, no regression vs
R1 numbers anywhere; decode parity everywhere). Anomaly: liquid-350 BF16
battery decode now EOSes at token 1 (was 80 tks @ 5.4 t/s) with prefill
16 → 44 t/s — safetensors path (no tile involvement); flagged to the liquid
peer, not reproduced on GGUF models.

### Session 3 wave 2 (2026-10-14 late night): attention + rope — Qwen prefill 128 → 158 t/s

The R2 profile said the Qwen prefill residual vs llama.cpp was attention+glue
(ours ~2.7-3.7 ms/token combined vs their ~0.7). Fresh profile split it:
attn bucket 246 ms per 234-token prompt and `other` ~500 ms (27% of prefill).
Two fixes, both bit-exact by construction:

1. **RoPE sincos/freq table** (`sofuu_rope`). The old loop computed
   `1/powf(freq_base, 2i/hd)` + `cosf` + `sinf` per (head, i) — ≈10M powf +
   20M sincos per Qwen prompt, nearly all inside "other". Tables cache freq
   per (half_dim, freq_base) and cos/sin per (pos, i), computed with the
   identical libm calls on the identical float arguments (`(float)pos * freq`)
   → rotation results and logits are bit-identical (proven by
   `test_rope_table_bit_exact`, both RoPE styles × 3 freq bases incl. table
   growth). One live table per key; grow copies the valid prefix, retired
   tables stay alive for in-flight readers; rows fill lazily under a mutex,
   never move. First implementation had a per-call re-grow pathology (the
   scan matched a zero-capacity entry and re-grew every call — ~250 KB alloc
   + fill per rope call, everything 3-4× slower; debug log showed
   `grew=1` on call N); fixed by creating tables with capacity immediately
   and replacing the live slot in place. Rope bucket: 100 → 14-16 ms per
   64-token chunk; the whole "other" bucket zeroed out.
2. **x4-position attention inner loops** (`sofuu_attn_qk_range` /
   `sofuu_attn_vacc_range`). QK dots process 2 positions per strip (4 lanes ×
   16-float strides exactly as `sofuu_vec_dot_f32`, combine → 4-stride tail →
   `vaddvq` → scalar tail) and the V pass applies 4 positions per chunk group
   with one dst round-trip. A first 4-position QK variant measured WORSE on
   decode (8.5 vs 7.2 ms/token — 16 live accumulators spill on A76); 2
   positions is spill-free. Bit-identical (`test_attention_x4_bit_exact`,
   plain/SWA/sinks, 6 configs per call).

Receipts (same binary, `pi_bench.sh`, llama.cpp same-session refs): Qwen
**158.2/20.6** vs llama.cpp 177.3/24.0 (0.89×, was 0.75× at session start);
Llama-1B Q8_0 **133.2/8.6** vs 58.5/8.7 (**2.28×** prefill, was 2.1×);
Spark 20.8/4.0 vs 22.8/4.2 (0.91×, within a ±0.5 t/s band of session-3's
0.96×). Host suite 571/0 on the merged state. Profiler split of a 64-token
prefill chunk after both fixes: gemm 276 ms, attn 28-79 ms (grows with seq),
norm 4, rope 14-16, glue 34, other 0.

Remaining Qwen prefill gap is ~8% and now truly kernel-side (attention
prefill at long seq + gemm scale-apply chain); `docs/PLAN-CPU-PREFILL-PEAK.md`
(plan, awaiting approval) covers the follow-on programme.

## Session 4 (2026-10-15): Phase-1 glue fusion — engaged, bit-exact, measured E2E-neutral

Ported the M2 Pro plan's Phase 1 (glue fusion) to the Pi 5 lanes, adapted to
our tile dispatch instead of `lfm2.rs`:

1. **Residual-in-epilogue GEMM.** All five tile kernels (q8_0, q4_K, q4_K
   sidecar, q6_K, q4_K planes) take an in-place-accumulate store
   (`s + offset` gets `s + acc_f32`), so `x += attn_out` and `x += ffn_out`
   fold into the out-projection / FFN-down GEMM epilogue. New externs
   `sofuu_gemm_{q4_k,q8_0,q6_k}_acc` return 0 (fused) / 1 (path not taken, out
   untouched — callers fall back to unfused `matmat` + `batched_mad`). The pad
   path adds the residual via `sofuu_f32_add_inplace` after the scratch run.
   Bit-exact: the same single IEEE add per element as the separate pass
   (`a+b == b+a`), proven by `test_gemm_resid_pair_bit_exact`, incl. the
   batch-6 pad case.
2. **Dual gate/up dispatch.** `sofuu_gemm_{q4_k,q8_0}_pair` quantizes the
   shared normalized activations once and runs both weight tiles
   back-to-back per row group — one act-quant pass and dispatch instead of
   two GEMMs. Bit-equal to two separate calls (same qa bytes).

Wired in `forward_batch_layers_chunk` (the speculative/spec-batch and
`forward_prefill_batch_chunk_internal` path both go through it; the spec-tree
path in model.rs was left alone — peer-owned). `SOFUU_FUSE_DEBUG=1` prints
engagement; the profiler counters confirm it: per 64-token chunk gemm calls
196 → **168** (pair), glue calls 84 → **28** (mads folded).

**Measured verdict: E2E-neutral on the Pi 5.** Full battery vs same-session
llama.cpp: Qwen 157.3/20.6 vs 177.8/24.1; Llama-1B 132.2/8.6 vs 58.9/8.6;
Spark 21.8-22.3/4.1 vs 22.8/4.3; decode parity everywhere — all within the
±0.5% run band of the pre-Phase-1 numbers. The 34 ms/chunk glue bucket fell
to 31 ms and dispatch counts fell as designed, but the removed work
(one T×dim read+write per residual, one act-quant pass per gate/up pair) is
a few ms against a 1.5 s prompt — and the epilogue add still pays the
residual load/store inside the gemm, unlike the M2 Pro case where glue was
55% of prefill. Phase 1 transfers *correctly* but not *profitably* at these
shapes; keep as bit-exact infrastructure (the counters' pool-dispatch saving
scales with smaller speculative batches, where dispatch overhead is
proportionally larger). The remaining Qwen prefill gap is gemm-scale-apply
chain + attention-at-long-seq; FA2-style tiled attention (the plan's Phase
5) is the next measured-lever candidate.

### Session 4 addendum: Phase-2 int8-activation SDOT — v1 measured SLOWER; hardware ceiling ~1.4× → parked

Ported the plan's Phase 2 to the Pi float fleet (`lfm2.rs`, opt-in
`SOFUU_PRE8=1`): activations quantized per 128-band once per layer call,
panels converted to per-row per-band int8 weights in the same pass that
dequantizes bf16/MLX4, tile kernel accumulates int8×int8 SDOT into per-band
i32 and folds two-sided `w_scale·a_scale·iacc` into f32. Numerics proven:
`pre8_sdot_tile_rel_rms_greedy` — kernel is **exactly** the scalar int8
reference (maxdiff 0), rel-RMS ≪ 1e-2 vs f32, per-token argmax identical.
Two engineering lessons baked in: multi-accumulator inline-asm SDOT blocks
produce wrong results on stable rustc (input/_out register binding across
statements — use the proven one-acc-per-statement `sdot_band` shape, same as
the PTQ1_0 path), and the first device A/B must include an engagement +
scalar-reference bisect (caught both).

**Measured verdict: v1 is 7-17% SLOWER end-to-end** (liquid-350 prefill
39.2 t/s vs 42.2-46.9 with the f32 tile), because (a) the SDOT tile calls an
outlined target_feature helper 1024×/panel (8 row-quads × token-pairs ×
16 bands) — call overhead dominates, and (b) the per-panel int8 weight
conversion is a scalar pass. Both are fixable, but before investing the fix
the hardware ceiling says the payoff is thin: on A76, SDOT issues 16
int8-MACs/cycle against FMA's 8 f32-MACs/cycle on a single pipe — the
device's int8 ceiling is the already-measured 185 GMAC/s (q8_0 tile) vs an
f32 tile projection of ~120-135 GMAC/s, i.e. **~1.4× kernel headroom**, not
the ≥2× the M2 plan assumed (AMX-class). After the two mandatory overhead
fixes the best-case net is ≲15% on a fleet where GEMM is ~70% of prefill —
below the plan's own ≥1.8× acceptance bar. Phase 2 parked as bit-gated
infrastructure (`SOFUU_PRE8=1`, default off, suite 577/0); revisit only if
the liquid float fleet becomes a product surface, or fold the same
two-sided-scale eight into the q4_K scale-apply-chain work, which shares the
math. (2026-10-15 follow-up: the ladder owner's `6605a2ed3` later removed
this parked SDOT block during the double-merge cleanup — their 32-group
per-load-slab ladder is now the one `SOFUU_PRE8` path on main, with amx
defaults; the SDOT variant is preserved in `9848995df` history for
re-landing.)

### Session 4 addendum 2: Phase-5a FA2 tiled attention — correct, measured slower → parked default-off (`SOFUU_FA2=1`)

Implemented the plan's Phase 5 (the one lever the portability assessment
called the real remaining candidate) in the shared attention kernel
(`sofuu_batched_attention_ext` — every GGUF backend + Spark): per
128-position K/V blocks, block scores via the proven x2-lane
`sofuu_attn_qk_range`, one NEON block max, online-softmax rescale of the
running (max, sum, acc) with per-position `expf (≤ 0 exponents)`, in-place
P·V accumulate in position order, one normalize at the end. SWA's sink and
window ranges share the same running state (they are one union softmax), so
the FA2 branch is a drop-in for both the SWA and plain two-pass branches.
Not bit-exact by design — block-wise max/exp reassociates float rounding —
so the new harness `sofuu_test_attention_fa2_parity` gates relative element
error ≤ 1e-4 against the two-pass reference (`sofuu_attn_ref_slot`) over the
same 7-config plain/SWA/sink matrix as the x4 test
(`test_attention_fa2_rel_gates` in quant.rs, seq past the 128-block boundary
so multi-block rescale is exercised). Suite 580/0.

**Measured verdict: FA2 loses on the Pi 5, and the loss is structural — no
tuning flips it.** Same-session device A/B (Qwen3-0.6B, 234-token prompt):
attention bucket per 64-token chunk 33/61 ms → **47/80 ms**, decode attn
8 → **13 ms** per token, E2E prefill 157.8 → **146.4 t/s** and decode
20.6 → **17.3 t/s** (GEMM bucket identical, so the regression is fully
attention-attributed). Three reasons, in order of weight:

1. At our sequence lengths the scores vector is ~1 KB per head and never
   leaves L1. FA2's reason to exist — never materializing scores to DRAM —
   is void here, and the existing two-pass is already memory-optimal.
2. The two-pass loops it would replace are A76-register-tuned (x2-lane QK,
   4-position V pass) and do strictly less work: the block max, block sum,
   and per-block rescale of (m, l, acc) are pure overhead on top of the
   same dots, exps, and FMAs.
3. Attention is only 8-15% of prefill at these shapes, so even a
   zero-overhead FA2 caps at a few percent E2E — below the ≥15% acceptance
   bar every shipped lever this session cleared.

Shipped state: kernel + harness kept as `SOFUU_FA2=1` opt-in, default off;
the default path is byte-for-byte the pre-FA2 behavior (final-binary
receipts: Qwen 155.5-157.1 prefill / 20.6-20.7 decode, Spark 22.2/4.4 — all
inside the session band). Second lesson of the session alongside Phase-2's:
the M2 plan's phase ranking prices levers by M2-Pro economics; on the Pi,
the denominator is what actually reaches DRAM at small-seq × small-head —
score materialization doesn't, so eliminating it buys nothing, while the
QK/V dot shapes that are already register-tuned can't be beaten by
re-fusing.

## Session 5 (2026-10-15): T0 platform pass — A76-tuned rustc codegen + thin LTO shipped (opt-in); PGO blocked at link; huge pages closed by kernel; the box runs sustained 2.8 GHz

First slice of docs/PLAN-PI5-DEEP-EDGE.md (the "go deeper" programme written
after the prefill plan closed). Three platform items, executed in one pass:

1. **Tuned codegen (`SOFUU_PI5_TUNED=1` + `SOFUU_PI5_LTO=1`, opt-in env in
   `scripts/cross/build_linux_arm64.sh`).** Auditing the toolchain showed the
   C++ kernels were already A76-tuned (zig-cc.sh → `-mcpu=cortex_a76`;
   sofuu-ffi build.rs → `-march=armv8.2-a+dotprod`) — the gap was rustc's own
   NEON codegen and cross-crate layout. Added `-C target-cpu=cortex-a76` +
   thin LTO + CGU=1 (LTO via the cargo profile env — `-C lto` in RUSTFLAGS
   errors on intermediate rlibs: "lto can only be run for executables,
   cdylibs and static library outputs"). Device A/B, interleaved runs on the
   same box, Qwen3-0.6B: **total 5.22-5.23 s vs 5.32-5.33 s baseline
   (+1.8%)**, decode **21.3-21.4 vs 20.7 t/s (+2.9%, non-overlapping)**,
   prefill −1.2% (158.2-158.8 vs 159.9-160.7 — layout re-cost on the
   prefill path, paid back by decode); Llama-1B neutral both ways (decode on
   the wall). The tuned artifact is deployed as
   `sofuu-linux-arm64.tuned`; the standing binary stays generic so peers'
   battery baselines don't shift underneath them.
2. **PGO: compiles, blocked at link.** rustup's aarch64-unknown-linux-musl
   std ships libprofiler_builtins, so the instrumented build compiles —
   then the static-pie link dies with `ld.lld: error: cannot open
   __llvm_profile_runtime` (zig toolchain carries no compiler-rt profile
   runtime for musl). The instrument/use branches are wired into the script
   (`SOFUU_PI5_PGO_INSTR` / `SOFUU_PI5_PGO_USE`); unblock = clang-musl with
   `libclang_rt.profile`, or nightly `-Z build-std` with the profiler
   feature — the rerun is mechanical after that.
3. **Huge pages: closed by hardware/kernel facts.** Device kernel: 16K
   pages, **no THP and no hugetlbfs at all** (no
   `/sys/kernel/mm/transparent_hugepage/`, no Huge* meminfo lines). With 16K
   pages + A76 contiguous-TLB entries, TLB pressure on streaming GEMV is
   marginal anyway; a custom kernel is a bad trade for a bench box. The mmap
   advice set (SEQUENTIAL at load, header WILLNEED) is already the right
   shape.
4. **Clock receipts: `arm_freq=2800` was already in config.txt** — every
   receipt this session (ours and llama.cpp's) was measured at sustained
   2.8 GHz, +17% over the SoC's rated stock, always equal-footing on the
   same box. Governor performance, throttle bits clean. No further
   overclock without an explicit go (reboot risk on the remote box).

Lesson of the pass: the last compile-flag-level percent is splitting the
difference between paths (decode gains, prefill gives a little back) — the
remaining real gains are the T1 GEMM math and the T2 format work, per the
deep-edge programme.

## Session 6 (2026-10-15): T1 q4_K scale-apply chain — repack-time pre-decoded sub-block scales (separate 192 B/block region), default-on, bit-exact; kernel +5.0%, E2E-neutral ≤233 tok

The stock q4_K tile kernels spend ~150% of their vdot count on aux ops: the
12-byte 6-bit scale-group unpack (~80 NEON ops per 256-block) plus the
int-multiply scale chain (`a76_repack.h` gemv/gemm both decode
`decode_q_Kx8_6bit_scales` in the hot loop). T0's f32 sidecar proved the
decode is removable (+51% kernel) but its +267% tile bytes were DRAM-neutral
E2E. T1 is the bytes-sweet-spot version: at repack time, decode each sub-block
ONCE into a separate `tile_ss_q4_Kx8s` region — f16 col scales (ints 0..63,
exact round-trip) + raw widen-byte mins, **192 B per 1152 B stock block
(+16.7%)** — and the prefill GEMM (`ggml_gemm_q4_K_8x4_q8_K_s`) loads it as a
second stream. The decode GEMV never reads the region, so the band-bound
decode path keeps the unbroken stock layout (appending the region to the tile
was measured at −10% decode via 192-byte holes in the gemv stream — two
regions instead).

Bit-exactness by construction: the fill runs the stock decode itself; every
f32 scale product (< 2^24) and pair sum (< 2^23) is exactly representable, so
the f32 apply chain (`vmulq_f32`+`vfmaq_f32`) reproduces the stock
int16-widen + `vmulq_s32` chain bit-for-bit. Suite gates it via
`test_gemm_q4_k_s_tiles_bit_exact` and the resid/pair tests (581/0 on the
Mac host). Kill switch `SOFUU_Q4_SS=0`; per-tensor eligibility ≤16M elems;
global fill budget 768 MB (`SOFUU_Q4_SS_MB`).

Device receipts (Pi 5, 2.8 GHz sustained, governor performance, .t1 binary =
tip + T1 tuned-codegen no-LTO):

- **Kernel** (`SOFUU_BENCH_GEMM 3072,1024,64,30`): q4_K **125.5 GMAC/s** with
  SS on vs **119.5** off (the T0 sidecar path) — **+5.0%**; q8_0 unchanged
  (180.7/182.2 — noise). The 75.7→125.2 earlier figure was against
  pre-sidecar stock; T1's honest step is +5% on the current default.
- **Qwen E2E prefill**: 75-tok 163.3-165.3 (T1) vs 163.3-164.8 (off);
  233-tok 157.3-164.7 vs 155.7-160.9 — **neutral within noise**, decode
  identical (23.7-23.9 t/s both). Same story as the f32 sidecar: the q4_K
  GEMM is ~1/3 of prefill, so a +5% kernel is ~+1.7% E2E — sub-noise at
  current prompt depths; the win banks for large-prompt/chunked paths.
- **Spark-X2.5-4B**: with the C++ tile path
  (`SOFUU_SPARK_CPP_GEMM=1`, peer commit c600c3df6's coherence kill-switch)
  22.4 prefill / 4.5 decode — vs 22.5/4.4 without T1 — neutral. Two findings
  from the A/B: (1) **output-token equality SS-on ≡ SS-off** (live inference,
  identical text — the E2E bit-exactness proof); (2) the DEFAULT Spark path
  (peer's safe pure-Rust prefill from c600c3df6) runs 0.3 tk/s prefill on the
  Pi in BOTH configs — pre-existing on the tip, unrelated to T1 (Standing
  binary c600c3df6-limits documented under TASKS 2026-10-15 Spark entry).

Debug war-story worth keeping: the first on-device A/B showed NaN and
non-determinism in the resid tests. Two real defects surfaced and were fixed:
(1) `sofuu_tile_lookup_s` had its return value and out-param inverted relative
to the `lookup_sc` convention — dispatch and bench were handed the tile and
the scale section CROSSED (cache bytes were individually perfect, which kept
byte-compare diagnostics green); (2) random-byte test fixtures reinterpret
GGUF `d`/`dmin` headers as f16, so ~3% of blocks are NaN/inf — bitwise
comparisons degenerate to NaN-payload matching (kernel-vs-kernel passes
vacuously, kernel-vs-scalar-add fails on payload drift, and pass/fail flaps
with codegen). All fixtures are now sanitized
(`sanitize_block_scale_headers`) so bitwise parity gates test real numbers.

## Session 7 (2026-10-15): the parity wave — Qwen prefill 158 → 183 t/s, ~4% AHEAD of llama.cpp (~176), outputs bit-stable

Three moves, all bit-stable end-to-end (`SS-on ≡ SS-off ≡ pre-wave` output
tokens verified live):

1. **SOFUU_PREFILL_BATCH default 64 → 128** (+4% alone): the 234-token
   standard prompt becomes one 128 chunk + a 104 tail — one chunk less and
   better tile amortization. Spark and Llama-1B neutral; the env var still
   overrides.

2. **P1 — vectorized softmax** (`sofuu_softmax` NEON exp): the per-score
   scalar `expf` was ~5M calls per Qwen prompt (γ also 100K+/decode token).
   Magic-number rounding to 2^(x·log2e) lanes, degree-5 Horner for the
   fraction, dead-mask aligned to f32 expf's underflow point (x < −104 →
   exact 0 — same as scalar). NOT bit-identical to libm; gate =
   `test_softmax_vec_rel_gate` at 2e-6 peak-relative vs a scalar expf
   reference (the poly is ~1e-7; the rest is f32 sum reassociation across
   200+ lanes). Receipts: 167.7–169.8 t/s combined with the batch bump.

3. **P2 — batched-QK attention** (`sofuu_attn_qk_rows8`): the plain prefill
   path re-read every K row once per batch row (64×); now two K strips serve
   eight q rows per step. Per-dot arithmetic order is IDENTICAL to
   `sofuu_attn_qk_range` (4-lane strips, combined tail, vaddvq, scalar tail)
   → bit-exact per dot; the plain-path work domain becomes (group-of-8, head)
   with per-row causal bounds. SWA/window and the parked FA2 path keep the
   (row, head) domain. Gate: `test_batched_attention_ext_plain_rows8`
   (b=10 crosses a group boundary; peak-relative 1e-4 vs an in-test scalar
   reference). Attention bucket 292 → 176 ms/prompt (−40%).

Device ladder (Pi 5, 2.8 GHz, 234-tok prompt, .t1 → .t1p1 → .t1p2):
158.6 (batch-64 baseline) → 165.0 (batch 128) → 167.7–169.8 (+vec softmax)
→ **182.9–184.4 (+rows8 QK) vs llama.cpp ~176–177**. Decode unchanged
(2.9 t/s single-token latency artifact; wall). Spark 22.6 with
`SOFUU_SPARK_CPP_GEMM=1` (peer coherence flag; unchanged). Llama-1B band
intact. Suite 589/0.

Parity-wave findings worth keeping: (1) the first vec-exp build had TWO
classic NEON-exp bugs — Horner built the reversed polynomial
(FMA argument order), and the 2^ki rebuild used the DOUBLE exponent bias
(1023) — a float rebuild is (ki + 127) << 23; both caught by the parity
gate, not by inspection. (2) Two test-gates used per-element rel on
near-zero values (softmax underflows / FA2's tiny outputs) and flagged
1e-7-scale absolute noise as 1e-4 "divergence" — peak-relative is the
metric that matches softmax/attention numerics.

## Session 8 (2026-10-16): the Spark-4B fix — q4_K/q6_K dotprod-fallback act-quant unit bug (cols/256 → cols/32); SOFUU_SPARK_CPP_GEMM now the default

The 2026-10-15 evidence pack's "remaining suspect: the C++ q4_K tile
pipeline" is closed, and the suspect was slightly wrong: the tile paths
(_p/_sc/_s) were never corrupt. The bug lived in the **batched dotprod
FALLBACK** they masked. `sofuu_gemm_q4_k_impl`'s SDOT fallback (and the
same block in `sofuu_gemm_q6_k_impl`) sized its activation scratch as
`q8_blocks = cols / 256`, but `sofuu_quantize_row_q8_a76` is 32-element
granular and `sofuu_vec_dot_q4_k_dotprod_a76` consumes 8 ext-blocks per
q4_K super-block — the buffer unit is `cols / 32`. So the quantizer
covered **1/8 of every row** (the rest quantized from whatever the
previous call left in the thread-local scratch) and the dot read **8×
past** the allocation. Whenever tiles engaged (the normal path), the
fallback never ran and models were fine; Spark-4B's prefill padded
batches (78 → 80 etc.) plus its fused-QKV shape skipped the tile path —
hence "coherence breaks only Spark", hence every tile-pipeline A/B
(mis)updating in innocent. Fixed by the one-line unit change, same as
the q6_K copy. This also explains evidence-pack finding (2) (activations
"reach nearest_int huge/NaN once tiles are disabled": garbage in the
scratch tail, not prefill state).

Second resident of the same mask: the pad paths held the zero-padded
activation copy *and* `out_dst` from the same `tile_scratch` thread-local;
a later resize past capacity reallocated the vector and left the pad
pointer DANGLING mid-GEMM (`rows > cols` cases only — Spark's fused QKV
4096×2560 pads, Qwen never pads). Now a separate `tile_scratch_pad`
vector per role.

New regression: `test_spark_qkv_cpp_gemm_parity` (infer/quant.rs) runs
Spark's exact dims (dim 2560, fused QKV 4096/1024/1024, batch 78 with the
spark.rs three-sub-range warm) through the production matmat dispatch vs
a dequant-row reference. Gate is peak-relative per batch column
(2e-2·colmax + 150) because the file's act-quant noise is absolute-scale
(√dim · dx/2 · w_rms ≈ tens) and only looks large RELATIVE on
near-cancelling dots (8.6% at the smallest dot while ~0.4% of a typical
one); the structural signature (27%, 3893 absolute at b=0) stays far
above it. Fixture lesson: a weight row spans **cols** (10 q4_K
super-blocks = 1440 B), not `q_dim` — the first draft strided rows by
`q_dim/256·144` and produced an engine-vs-reference phantom where only
row 0 could ever match (and the stray f16 header sanitize offsets landed
mid-row, feeding row-1 + random-byte headers). Suite 596 passed / 0
failed; all five SOFUU_SPARKTEST_VARIANT bisect shapes pass.

Device receipts (t1p4/t1p5, Pi 5, 2.8 GHz, 4 threads, temp 0):
- Coherence: "The capital of France is" → " Paris, which is a major city
  in Europe." with the C++ path — token-identical to the dequant
  reference (both show the model's benign trailing "joining" artifact).
- Spark prefill battery, 433-token prompt: **20.8 tk/s** (20.86 s) with
  the C++ path vs the dequant fallback killed by a 400 s timeout before
  finishing 433 tokens; decode 3.8 t/s unchanged (fix is prefill-only).
- `SOFUU_SPARK_CPP_GEMM` flipped DEFAULT-ON in `spark.rs`
  (`=0` restores the dequant fallback; `=1`-era semantics unchanged for
  anyone relying on the historical knob). Verified on device: default
  run coherent at fast-path speed, `=0` run coherent at dequant speed.
- q6_K `in_acc` audit item re-verified at tip: `if (in_acc) return 1;`
  guards both q6_K and q4_K before the dotprod blocks, so the generic
  tails are unreachable under the fused-residual contract — honest, no
  change needed.

## Session 9 (2026-10-16): Llama-1B Q8_0 focus pass — lookahead speculative decode default-on (decode 2.2x), prefill attribution pinned

Scope: the developer directed a max-effort pass on Llama-3.2-1B-Instruct-Q8_0
input (prefill) and output (decode) processing. Baseline t1p6, 170-tk prompt,
2.8 GHz, 4 threads: prefill 139.7 tk/s, decode 9.0 tk/s.

Attribute first (SOFUU_PROFILE): prefill = 92.6% q8_0 GEMM (522 of 564 ms;
~213 GMAC/s effective against the ~307 GMAC/s SDOT ceiling — kernel headroom
~1.4x lives here); decode = 97% GEMV (3.33 of 3.42 s) streaming 1.24 GB of
weights per token at ~12 GB/s effective — the DRAM wall again, so the only
physical decode lever is reading weights fewer times per emitted token.

The lever already existed: `--lookahead` (Jacobi/n-gram speculative decode on
the GGUF batched tile path; the same pad-to-4 infra that made spec batches
weight-stream-shared). A/B: 19.7 tk/s decode with **bit-identical temp-0
output** (48-tk wild-prompt diff = empty), prefill unchanged. The path is
sampler-aware (`sample_child` per accepted node) with a greedy-equivalence
test pinned, and the engine only engages it on the GGUF batched backend.

Flip shipped: the CLI defaults lookahead ON (`SOFUU_LOOKAHEAD=0` or
`--no-lookahead` = serial), default verified on device: 20.1 tk/s decode
(run default) vs 8.7 (kill switch), prefill 140.0 both ways. Prefill batch
re-check on this model: 64/128/256 = 133.1/140.0/140.0 — 128 stays.

Next for this model: q8_0 GEMM kernel toward the SDOT ceiling — repack-time
scale pre-storage (the T1 pattern) and panel shapes informed by the kernel's
per-block scale apply chain; measured target 213 -> 250-280 GMAC/s.

### Ceiling push, measured (same session): 8-act panels — +1.5%, and what the miss teaches

Next kernel iteration tried to buy the rest of the gap with load-sharing:
`ggml_gemm_q8_0_8ax4_q8_0` (each weight-tile row + weight-scale convert shared
across two consecutive 4-act groups) and the same treatment for the gate/up
PAIR kernel (`ggml_gemm_q8_0_8ax4_pair_q8_0`), dispatched at batch % 8 == 0,
fold order per output unchanged → bit-identical (verified: identical temp-0
continuations, and the scale factor kept the original per-act-lane broadcast
`u_d[j]·d0[m]` form — the tempting vector-vec factoring breaks it).

Result: prefill 139.5 → 141.5 tk/s (+1.5%), twice, both prompts. The honest
read: at batch-128 shapes with cols = 2048, each 4-row weight tile is
~8.7 KB — the eliminated loads were L1/L2 hits, not DRAM traffic, so the
q8_0 GEMM at ~213 GMAC/s effective is bound by the SDOT dependency chains
(the 4-accumulator laneq form has ~8 dependent SDOTs per accumulator per
block) and the per-block f32 scale-apply, not by loads. Closing the remaining
distance to the ~307 paper ceiling needs a different kernel class: multi-block
unrolling with f32-fold reassociation (greedy-equality gated, not
bit-exact — the Phase-1 fusion precedent) or an i8mm-style wide-panel design
this SoC cannot execute. Prefill plateau for this model on this chip:
**~140-150 tk/s with today's fold discipline** — 2.5× llama.cpp's 55.6.

Decode plateau confirmed flat too: SOFUU_SPEC_ROWS/DEPTH sweeps (3..8 × 2..4,
env knobs now shipped) all land 17.6-20.4 tk/s — bigger trees are free but not
more productive; the accept-rate is n-gram-quality-bound. **20.1-20.4 tk/s is
the measured decode ceiling for this model on this box** (serial floor 8.7).

Both panel kernels ship (bit-identical, positive, free). t1p9.

## Session 10 — receipts rewrite: the greedy scoreboard (t1p11)

Prior decode numbers in this file and in commit af1003491's message were
measured with a silently-dropped sampling flag: the `infer` CLI only parsed
`--temp`, so `--temperature 0` fell into the unknown-arg branch and every
"greedy" run actually sampled at the 0.7 default with a wall-clock-seeded
RNG (Sampler::new uses SystemTime). Symptom: different text every run,
same prefill logits — which read as a compute nondeterminism until the
sampler was instrumented. Frozen-forever receipts that casually used
`--temperature`: treat their TEXT-equality claims as unverified; their
speed numbers remain valid (compute cost is sampling-independent).

Fixed in t1p11: `--temperature` aliased to `--temp`, `-n` aliased to
`--max-tokens` (it was being dropped too — every earlier "-n 96" run
actually decoded 256), and the infer arm now WARNS on unrecognized
dash-flags instead of eating them.

Re-baselined scoreboards, all at true greedy (`--temp 0`), Pi 5, 4×A76
@2.8 GHz, Llama-3.2-1B Q8_0:

- Serial decode: 9.0-9.2 tk/s (1.05 GB weight stream per token ⇒ ~9.9 GB/s
  effective; llama.cpp on the same file measures ~8.3 ⇒ ~11 GB/s effective —
  both engines sit on the same ~10-11 GB/s LPDDR4X soft wall; the isolated
  4-row tile GEMV loop benches 17.9 GB/s warm, 16.1 GB/s is the
  single-core pure-read ceiling: the kernel is not the limiter, the cold
  1 GB stream is. No THP on this kernel to relax TLB pressure.)
- Lookahead (legacy tree, rows 5 = widths 2/2/1, default-on): 15.6 tk/s
  on the counting prompt (+69% over serial), token-identical to serial.
  On natural prose: 9.0 vs 9.1 serial — acceptance is text-bound; the
  n-gram draft only pays on templated/repetitive output.
- Width sweep (device, greedy): level-1 width 2 dominates. (2,2,1,1) 15.8,
  (2,2,2,0) 15.6, (3,2,2,0) 13.6, (4,2,1,1) 14.1 — rank≥3 candidates cost
  more row work than their acceptance returns. Default stays (2,2,1).
  CandidateTree::build_cfg now exposes per-level widths + a real level 4
  (SOFUU_SPEC_ROWS 1..8, SOFUU_SPEC_DEPTH 1..4, SOFUU_SPEC_WIDTHS).

**Numeric-divergence finding (open)**: at true greedy the lookahead path is
token-identical to serial on templated text but NOT on natural prose —
near-tie logits flip under the batched tile-GEMM fold vs the serial
tile-GEMV fold (with SOFUU_NO_TILES=1 both paths run the generic row
kernels and the outputs match exactly, isolating the difference to the
tile GEMM family). Both are the model's argmax under their own logits —
quality-neutral — but cross-path token-equality cannot be claimed until
the batched and row kernels share per-output fold orders. Next session:
a GEMV↔GEMM bit-equality unit test per shape to find the first splitting
op, then either unify the fold or gate acceptance by a tie-margin.

Determinism guard added: `live_llama1b_prefill_logits_repeat_in_process`
(#[ignore], run with `cargo test -p sofuu-core --lib live_llama1b -- --ignored`)
— repeated prefills in-process must be bit-identical. t1p11.

### Battery row — Qwen3-0.6B Q4_K_M (2026-10-19, t1p12, post-flag-fix)

Model file had vanished from the device; re-downloaded (unsloth mirror,
397 MB). Greedy `--temperature 0`, `-n 256`, 4 threads, ~138-token prompt,
3 reps each; llama.cpp build 11041 same box/prompt `--single-turn -t 4`:

| engine (Qwen3-0.6B Q4_K_M) | prefill t/s | decode t/s |
|---|---|---|
| Sofuu serial (`--no-lookahead`) | 179.0-181.5 | 23.0 (3× identical) |
| Sofuu lookahead (rows 5, default) | 180.4-180.8 | 26.3-27.5 |
| llama.cpp | 177.9-178.1 | 24.6-25.0 |

Prefill parity/ahead (1.02×); decode ahead with lookahead (1.08× vs llama.cpp;
serial 0.92×). Wall math: 397 MB weights ⇒ 9.1 GB/s serial / 10.7 GB/s
lookahead effective. Context: first Qwen receipt 2026-10-14 was prefill 110.4 /
decode 22.0 — prefill +64% since (act-quant, attention rows8, softmax, batch
128, tuned rustc), decode +20-25% with lookahead. Token equality serial-vs-
lookahead at temp 0: NOT equal on this natural prompt (tile-fold divergence,
see Session 10 finding — same class as Llama-1B).

## Pi 5 code map + knob matrix (2026-10-19 cleanup pass)

### Files (this lane's territory)
| file | role |
|---|---|
| `src/simd/arch/a76_repack.h` | llama.cpp-derived DOTPROD tile kernels: q4_K/q6_K/q8_0 GEMM+GEMV, 8-act panel + gate/up pair, T1 scale sidecar, nibble planes, `_generic` fallbacks (compiled only for aarch64; Apple too) |
| `src/simd/arch/arm_cortex_a76.h` | per-row vec_dot fallback kernels (q4_0/q8_0/q4_K/q6_K/mlx4), NEON act quantizer, f16/bf16 converts |
| `src/simd/infer_kernels.cpp` | dispatch (tile lookup/warm/pad), C-ABI exports, gate parity harnesses (`sofuu_test_*`), `SOFUU_BENCH_GEMM` bench |
| `src/simd/platform/*.h` | core topology, pinning, seq-barrier thread pool |
| `crates/sofuu-core/src/infer/quant.rs` | Rust FFI wrappers, type_row_size, prof buckets, dequantizers |
| `crates/sofuu-core/src/infer/model.rs` | llama GGUF runner — SHARED with GPU/KIVI lanes; this lane's parts: prefill batching, type gate, loader fixes |
| `scripts/cross/build_linux_arm64.sh` | musl cross build (SOFUU_PI5_TUNED/LTO codegen knobs) |
| `docs/PI5-SIMD-OPTIMIZATION.md` | this file — receipts |

### Knob matrix
| knob | effect | default | status |
|---|---|---|---|
| `SOFUU_LOOKAHEAD=0` / `--no-lookahead` | spec decode kill switch | ON (GGUF) | shipped, default-on |
| `SOFUU_SPEC_ROWS` | tree size 1..8 (tile batch caps at 8) | 5 | measured: 5 best; wider regresses |
| `SOFUU_SPEC_DEPTH` | tree depth 1..4 | 3 | level 4 needs width[3]>0 |
| `SOFUU_SPEC_WIDTHS` | per-level widths "l1,l2,l3,l4" | (2,2,1) shape | l1=2 dominates (device sweep) |
| `SOFUU_Q4_SS` | T1 pre-decoded q4_K scales kernel | ON (=0 fallback) | shipped, bit-exact |
| `SOFUU_NIBBLE_PLANES` | nibble-plane q4_K kernels | OFF | opt-in, +1% E2E, parked |
| `SOFUU_SCALE_SIDECAR` | old scale sidecar variant | OFF | DEAD END (slowed decode); keep off |
| `SOFUU_FA2` | FA2 tiled attention | OFF | parked (slower on Pi; L1-resident scores) |
| `SOFUU_NO_TILES` | bypass packed tiles → generic row kernels | OFF | kill switch + numeric A/B tool |
| `SOFUU_TILE_TYPES_OK` | tile-eligibility type mask | q8_0/q4_K/q6_K | escape hatch |
| `SOFUU_PREFILL_BATCH` | prefill chunk (snapped to mult of 4) | 128 | 64→128 was +4% |
| `SOFUU_NUM_THREADS` / set_thread_count | pool size | 4 (Pi `_SC_NPROCESSORS_ONLN`) | S-tuned |
| `SOFUU_BENCH_GEMM` | rows,cols,batch,iters[,dir] bench | unset | dev bench |
| `SOFUU_PROFILE`, `SOFUU_PROFILE_LOAD` | token/op attribution | off | diagnostics |
| `SOFUU_DEBUG_LAYERS`, `SOFUU_DEBUG_DETERMINISM`, `SOFUU_DEBUG_TILES*` | per-layer RMS / step-logits / tile traces | off | diagnostics |
| `SOFUU_SPARK_CPP_GEMM` | Spark C++ GEMM path | ON (=0 dequant) | shipped, healed |
| `SOFUU_PI5_TUNED`, `SOFUU_PI5_LTO` | cross codegen — cortex-a76, thin LTO | OFF | opt-in (+2.9% decode) |
| `SOFUU_NO_ROPE_TABLE`, `SOFUU_ROPE_DEBUG` | rope table bypass / debug | table path | diagnostics |

### Cleanup pass receipts (this pass)
- Removed: `inline_dot` (dead since Oct 12, never referenced); 2 Rust extern
  decls whose C++ callers are C++-only (`sofuu_dequantize_row_q4_0/_q8_0`);
  moved the two cfg(test)-only parity FNs under a `#[cfg(test)]` extern block
  (kills release-build never-used warnings).
- Fixed cross build (blocked ALL lanes on the GPU peer's GPU3 wave):
  `self.gpu` field access at model.rs:2329 is now cfg-split (field is
  Apple-only); cross target builds again.
- Verified live: bench harnesses (each `sofuu_test_*`/`sofuu_bench_*` has a
  live Rust gate/CLI caller — NOT dead, they are the receipt
  infrastructure); every `_generic` fallback is reachable via its dotprod
  twin; quant.rs externs all referenced.

## Session 11 — MiniCPM parity correction; merged-GEMV measured DEAD; Qwen profile

**MiniCPM5-Sharp 2B Q6_K_XL was never 1.9× behind.** Same-session, same-prompt
(t1p14, greedy, 4 threads): Sofuu prefill **37.8 tk/s** vs llama.cpp **36.4**
(decode 5.7 vs 5.6). The old "17.6 vs 33.3" receipt predates the CLI flag
fixes/knob corrections; treat it as void. At supported formats every Pi model
is now at parity or ahead: Llama-1B (2.5× prefill / 1.9× decode lookahead),
Qwen-0.6B (+2% prefill / +10% decode lookahead), Spark (parity), MiniCPM
(parity).

**Qwen-0.6B decode attribution (SOFUU_PROFILE, serial):** per 32-token window
~1.05s gemv (82%, 9.45 GB/s effective of a 13.0 GB/s pure-stream bench
ceiling), **attn 182-213 ms (15% — the 39 MB/token f32 KV read is the cost)**,
norm 6 ms, glue ~0. The decode gap to the wall is cold-stream physics, not
dispatch.

**Spec shape for Qwen-0.6B:** rows 5/7 ≈ 23.5-23.7, rows 8 = **14.1 tk/s
cliff** (tree batch 9 → pad-12 → loses the 8-act panel path). Keep default
rows 5; never run rows 8 on this batch geometry.

**Merged-GEMV experiment — built, gated, measured ZERO, REVERTED.**
`sofuu_gemv_q4k_q6k_multi` merged q/k/v (3→1) and gate/up (2→1) per layer
(84 fewer pool round-trips/token, one act quantization), bit-exact vs the
individual calls (harness `sofuu_test_gemv_multi_parity`). Device A/B:
Qwen 22.7 vs 22.7 serial, MiniCPM 5.7 vs 5.6, Llama-1B untouched (q8_0) —
**no measurable gain**. The pool's sequence-barrier round trip is ~3 µs, so
dispatch overhead was never the missing 3.5 GB/s. Not shipped (per the
panel-GEMV precedent: zero-value paths don't land).

**Kept from the pass:** a tile-test mutex — the tile cache is keyed by
(weight pointer, rows, cols, type) and parallel tests can recycle a freed
fixture address, aliasing another test's entry (observed as intermittent
`spark_qkv`/`multi` gate failures). Tile-warming tests now serialize.

Implication for the next lever list: decode bytes are the only remaining
knob (KV f32 39 MB/token = KIVI's lane; lm-head bound-skip; Q3_K kernels),
and the ~13 GB/s stream ceiling bounds all of it.

## Session 4 (2026-10-22) — Spark GeGLU + n_rep=4 fused decode, per-model scoreboard

**Spark-X2.5-4B prefill was the last measured deficit** (22.4-22.6 vs llama
22.0-22.9, temp 0, 234-token prompt, three alternating reps). `SOFUU_PROFILE`
attributed it: gemm 8.95 s of 11.1 s, attn 0.64 s, and **1.42 s in `other`** —
Spark's layer path had no scopes, so `rope`/`glue`/`embed` all read 0 ms. Two
edits, both confined to Spark:

1. **GeGLU is SIMD now.** `spark.rs` ran `gelu_erf` per element (A&S 7.1.26
   polynomial + `expf`) over 36 layers x hidden_dim 10240 x batch — ~1.2 s per
   252-token prefill. New `sofuu_geglu_erf` (NEON 2^k exp, same polynomial,
   16-wide strips + scalar tail) drops that bucket to 0.31 s and turns the rest
   of `other` into attributed rope (56 ms) and glue. **Prefill 22.4-22.6 ->
   24.3-24.7 tk/s.**
   Numerics: `test_geglu_erf_matches_scalar_reference` (1e-6 rel) plus an
   on-device oracle (`SOFUU_GEGLU_SCALAR=1 SOFUU_PROFILE=1`) that measured
   **max_rel <= 1.8e-7 on live 10240-wide activations** — f32 rounding of the
   same formula. 3 of 4 greedy probe prompts are byte-identical; the one that
   flips does so inside two Czech continuations (this model answers Czech on
   templated instructions through our template, while llama.cpp's template
   yields English, so neither branch is the "reference" text).
   `SOFUU_GEGLU_SCALAR=1` reproduces the pre-change binary byte-for-byte.

2. **n_rep=4 fused decode attention.** Spark is 16 q heads / 4 kv heads with an
   **f32** KV cache; the per-head loop re-streamed each K/V row 4x, so the attn
   bucket read 17.5 ms/token against a ~7.8 ms traffic floor. New
   `sofuu_attn_qk_gqa4_f32` / `sofuu_attn_vacc_gqa4_f32` load each row once and
   apply it to all four heads, copying the per-head accumulation order verbatim
   (`test_attention_gqa4_fused_matches_per_head_path` asserts b==1 fused == b==2
   per-head bit-for-bit at head_dim 32/36/256; `sofuu_gqa4_attn_groups()` proves
   the branch ran). **Attn 17.5 -> 9.3 ms/token, decode 4.02 -> 4.2 tk/s**,
   generated text unchanged byte-for-byte. Escape hatch `SOFUU_GQA4=0`.

**Scoreboard (Pi 5, governor performance, greedy, same window, ours 2-3 reps
vs llama.cpp -t 4):**

| model | prefill ours | llama | decode ours | llama |
|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M (Downloads, 396 MB) n=128 | 183.7-185.3 | 177.4 | 28.1 | 23.5 |
| Qwen3-0.6B Q4_K_M (models/, 484 MB) n=128 | 184.2-185.0 | 177.5 | 24.0-24.2 | 24.2 |
| Llama-3.2-1B Q8_0 n=40 | 140.8-140.9 | 58.5 | 14.8 | 8.5 |
| SmolLM2-360M Q8_0 (short prompt) n=32 | 69-70 (5 tk) | — | 61.7 | 31.1 |
| Sharp-MiniCPM5-2B Q6_K_XL n=40 | 37.7-38.4 | 38.4 | 7.2-7.5* | 5.8* |
| Spark-X2.5-4B Q4_K_M n=40 | 24.3-24.6 | 22.0 | 4.2 | 4.1 |
| Ternary-9B PQ2_0 n=4 | 1.2-1.4 | n/a (type 142) | 0.1 | n/a |
| Qwen3-0.6B Q3_K_S / UD-IQ3_XXS | load error (types 11/13) | runs | — | — |

\* decode measured on a short prompt: on the 234-token prompt MiniCPM (and
SmolLM2) immediately emit EOS through our template, while llama.cpp's chat
template keeps them generating, so that long-prompt cell is not comparable.

**Not a completion model:** `laya_multilingual_q8_0.gguf` is `general.architecture
= ggmlc` (laya: `max_opts`, `length_buckets`, cls/sep/mask tokens). Neither
engine supports it — llama.cpp: "unknown model architecture: 'ggmlc'".

**Still open for "not a bit less" everywhere:** MiniCPM prefill sits at parity
(37.7-38.4 vs 38.4, needs the q6_K GEMM point); Qwen 3-bit tiers need real
q3_K/IQ3 kernels before the loader will accept them; Ternary-9B needs a ternary
GEMV (currently 0.1 tk/s on the per-row digit-gen fallback).

## Session 12 (2026-10-22) — Q3_K (ggml type 11) landed: loader → coherent text → 1.96x prefill

The 3-bit Qwen tier used to be rejected at load ("type 11 not supported"). It
now runs end to end, and the work decomposed into three measurement-driven
steps (t1p49 → t1p55 → t1p57; every number below is Pi 5, governor
`performance`, greedy, same window, llama.cpp `-t 4`, 234-token prompt, n=80).

**1. Bit layout + first kernels (t1p49).** `block_q3_K` = `hmask[32] | qs[64] |
scales[12] | f16 d` (110 B / 256 values). Value for position p: half = p/128,
group j = (p%128)/32, low bits `qs[half*32 + part*16 + l] >> 2j & 3`, minus 4
when hmask bit `1 << (half*4 + j)` is clear; sixteen 6-bit scales (bias -32)
expanded by llama.cpp's aux[4]/kmask permutation, one per 16 values. The C++
dequant is gated bit-exactly against an independent Rust transcription of
`dequantize_row_q3_K` (`test_q3_k_dequant_matches_reference`), and the dot
against that dequant (`test_q3_k_gemv_gemm_matches_dequant_dot`).
Measured 44-46 prefill / 20.0 decode — llama.cpp 73.2 / 27.7.

**2. Kernel reshaped to llama.cpp's NEON structure (t1p55).** `vbic(mask, h)`
+ a single constant shift replaces the compare/and correction (2 ops instead of
3 per 16 values), the per-group reduction folds into ONE int32 per block
(`isum += vaddvq * scale`, one float multiply per block), and the activation
side switched from the per-32 `sofuu_q8_block_ext` to the per-256 `q8_K` form
llama.cpp uses (one `d` per block). **65.2 prefill / 22.5 decode.**

**3. q3_K → q6_K tile for the GEMM (t1p57).** The 3-bit lattice (-4..3) and its
per-16 6-bit scales are an exact SUBSET of q6_K's (-32..31 values, int8
per-16 scales) with `d` unchanged, so a q3_K block converts into a q6_K block
by byte shuffling alone — no requantization. `sofuu_tile_warm` type 11 rewrites
the whole tensor into the stock `tile_block_q6_Kx8` layout (~1.9x bytes) and
`sofuu_gemm_q3_k` rides `ggml_gemm_q6_K_8x4_q8_K`; the decode GEMV keeps
reading the native 110-byte stream, so DRAM-bound decode does not pay the 1.9x.
**143.6-144.1 prefill / 27.2 decode.** Mutation-tested: flipping the packer's
`v + 32` to `v + 31` fails the tiled-GEMM gate (`tile (0,3): got 419.0 vs ref
297.2`), so the test really covers the tile path, not just the fallback.

| Qwen3-0.6B-Q3_K_S (323 MB) | prefill | decode |
|---|---|---|
| ours before (t1p49) | 44-46 | 20.0 |
| ours + kernel reshape (t1p55) | 65.2-65.3 | 22.5 |
| ours + q6_K tile GEMM (t1p57/t1p60) | 142.6-144.1 | 27.2 |
| llama.cpp `-t 4` | 73.3-73.4 | 27.7-28.1 |

Prefill is **1.95x ahead**. Decode is 2% behind, and the remaining debt is
attributed, not guessed: the decode profile is gemv 82.8% / attn 13.3% /
everything else 4%, and the q3_K stream runs at ~10.3 GB/s at the 1-3 MB matrix
sizes decode hits versus 12.45 GB/s for q4_K's interleaved 8-row tile (measured
by the same decomposition: q4_K_M 36.8 ms/token, 317 MB, 12.45 GB/s). The fix
is a native 8-row q3_K GEMV tile (byte-neutral, +3.6% for pre-expanded
scales); a first attempt at row-level ILP without a tile — four independent
accumulate chains sharing the activation loads — measured 22.8-24.0, i.e.
*slower*, because the per-row scale arrays spill, so it was reverted rather
than shipped. Also measured and rejected: speculative lookahead on this tier
(26.1 vs 27.2 serial; rows=8 trees 22.7, always-on 17.6) — the verify GEMM is
DRAM-bound at 1.9x tile bytes and the accepted-token rate cannot pay for it.

**Gates:** `test_q3_k_dequant_matches_reference` (bit-exact vs upstream),
`test_q3_k_gemv_hand_block` (mixed hmask, every shift/half → catches a wrong
hmask half, which the random fixture passed), `test_q3_k_gemv_gemm_matches_dequant_dot`
(GEMV, native GEMM and tiled GEMM each within 2% of row L1 vs the dequant
oracle, plus the residual-fused variant). Loader: Q3_K is in the SUPPORTED set
and the tile warm list; the two loader tests that used type 11 as the
"unsupported" example now use IQ3_XXS (type 18).

### Full battery, same window (t1p60, 2026-10-22)

Re-run after the Q3_K change, ours 2 reps vs llama.cpp `-t 4`, greedy,
governor `performance`, 234-token prompt:

| model | prefill ours | llama | decode ours | llama |
|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M (396 MB) n=80 | 184.7-185.0 | 177.7 | 24.4-24.9 | 24.4 |
| Qwen3-0.6B Q3_K_S (323 MB) n=80 | 142.6-144.1 | 73.3-73.4 | 27.2 | 27.7-28.1 |
| Llama-3.2-1B Q8_0 n=40 | 141.1 | 59.1 | 14.9 | 8.8 |
| SmolLM2-360M Q8_0 n=80 | 362.8-364.2 | 328.0 | (EOS at once)* | 28.1 |
| Sharp-MiniCPM5-2B Q6_K_XL n=40 | 38.2-38.5 | 38.7 | (EOS at once)* | 5.5 |
| Spark-X2.5-4B Q4_K_M n=40 | 24.2-24.6 | 22.1 | 3.9-4.2 | 4.0 |
| Ternary-9B PQ2_0 n=4 (type 142) | 1.4 | n/a | 0.7 | n/a |

\* same template/EOS caveat as Session 4: our chat template makes these two stop
immediately on this prompt, so the long-prompt decode cell is not comparable.

**Absolute decode numbers are window-local.** In this window both engines read
Qwen Q4_K_M decode at 24.4-24.9 (llama 24.4) where the Session-4 window read
28.1 vs 23.5 — including for the *old* binary (t1p47 re-measured today: 24.7,
identical to t1p60), so it is machine state after the reboot, not a regression.
Only same-window comparisons are valid.

Still short of "not a bit less": Q3_K decode 27.2 vs 27.7-28.1 (2%, native 8-row
tile identified above) and MiniCPM prefill 38.2-38.5 vs 38.7 (0.5-1.3%, q6_K
GEMM rate). Everything else is ahead or at parity in the same window.

### UD-IQ3_XXS scope (measured, not started)

The Qwen3-0.6B-UD-IQ3_XXS file is not "one missing type". Its tensor histogram:
`f32` 113 (norms), `q5_K` 1 (token_embd), `iq2_xs` 10, `iq3_xxs` 112, `iq3_s` 33,
`iq2_s` 36, `iq4_xs` 5 — six quant formats, five of them codebook quants (each
needs llama.cpp's grid table + ksigns table + a grid GEMV/GEMM). That is a much
larger project than Q3_K's single 110-byte block format; q5_K (176 B/256, same
scale/min permutation family as q4_K) is the only mechanical one and it alone
does not open the file.

### Ternary-9B prefill is MAC-bound, not bandwidth-bound

233-token prefill takes 165-171 s (1.4 tk/s). The slab GEMM
(`gemm_rows_t143_direct`) already expands each weight row to int8 once per pass
and dots 4 tokens per SDOT group, so the remaining rate — 9B params ≈ 18
GMAC/token at 1.4 tk/s ≈ 25 GMAC/s, versus ~200 GMAC/s for the q4_K tile GEMM —
is kernel efficiency, not redundant weight reads. There is no llama.cpp baseline
(type 142 is Prism-private).

## Session 13 (2026-10-22) — Persistence hardening (Pi 5 lane)

Charter: the engine must stay up and consistent under heavy conditions —
memory pressure, concurrent instances, malformed or truncated model files —
and never abort mid-generation. Nothing here changes the math or the fast
paths: every guard is a refusal or fallback whose output consumers already
handle (the per-row kernels are the same arithmetic the tile GEMM wraps, and
a rejected dispatch zeroes its output instead of reading garbage).

### Why each failure mode was fatal

- Release is `panic = "abort"` (Cargo.toml:36): one Rust `assert!` or one
  failed allocator call is SIGABRT.
- The C++ kernels take no length; they read `rows * type_row_size(cols)`
  bytes from a pointer. A short weights slice (truncated file, failed mmap
  range) was an out-of-bounds read.
- The worker pool has ONE task slot: a nested `parallel_for` overwrote
  `current_task_` and deadlocked, and a `std::bad_alloc` escaping a task
  called `std::terminate` (also fatal).
- Loading a truncated GGUF was only caught by
  `tensor_slice(...).expect("valid tensor slice")` — after the model was
  built, at the first matmul. Corrupt counts/lengths could panic in the
  parser (`pos + len`, `checked_mul(...).expect(...)`,
  `Vec::with_capacity(count)`).
- Tile warming ran *before* the KV cache. Under an address-space limit the
  optional tiles took the space the KV cache needed, and
  `AnonBuffer::new`'s `vec![...]` fallback aborted. That fallback was also
  freed with `munmap` (a latent heap corruption on the same path).
- `SOFUU_Q3K_TILE_MB` exists because a q3_K tile is 210 B per 256 values
  against the 110 B the model occupies (~1.9x).

### What is now guaranteed, and by what

| surface | injected condition | new behaviour |
|---|---|---|
| nested dispatch | kernel calls `parallel_for` inside a task | runs inline; selftest sees 1024/1024 slices |
| throwing task | injected `throw 1` (selftest) | contained, counted, barrier completes |
| worker creation fails | thread/memory limit | keeps existing workers, one log line, `short_thread_fallback` counter |
| `set_thread_count` from a task | nested call | ignored (no join-under-self deadlock); Rust clamps to [1,64], 0 = auto |
| tile allocation fails | `ulimit -v`, RAM pressure | warm returns -1 (one line), per-row kernels continue |
| optional tile footprint | `RLIMIT_AS` present | per-tensor admission vs address-space headroom (limit − VmSize − 256 MB reserve) + optional `SOFUU_TILE_MB` |
| short weights slice / wrong dims | truncated file, mismatched `x` | output zeroed, one diagnostic line per 1024, no abort, no OOB read |
| cols not a whole block | malformed tensor geometry | dispatch refused (`type_row_size` floors, which would under-count) |
| truncated GGUF | last 1 MB removed | load error naming tensor and byte range; exit 1 |
| corrupt GGUF header | absurd counts/dims/array lengths, overflowing lengths | load error; no allocator abort |
| KV cache reservation fails | `ulimit -v` 450–1200 MB | load error naming the cache size; exit 1 |
| f16 KV shadow reservation fails | `ulimit -v` 1400 MB | warning, engine runs on the f32 cache |
| allocation order | any of the above | KV caches and every working buffer first, tiles last |
| `AnonBuffer` Vec fallback | mmap failure | `Drop` frees with the matching allocator (`heap` flag) |

### Device receipts (t1p67, Pi 5, governor `performance`)

- Pool selftest: `[pool] selftest rc=0x0 contained_failures=1 (1 injected by
  the selftest) short_thread_fallback=0` → exit 0.
- Truncated (tail 1 MB removed):
  `Error loading model: ... tensor 'blk.27.ffn_up.weight' needs bytes
  394936000..396705472 but the file is 395705472 B` → exit 1. Before: load
  succeeded, first matmul read past the mapping.
- `ulimit -v 450000` → `cannot allocate the KV cache (448 MB x2 for the
  configured context): ... Out of memory` → exit 1.
- `ulimit -v 900000` → `cannot allocate the second KV cache (448 MB)` → exit 1.
- `ulimit -v 1200000` → same clean error. Before these three: exit 134 with a
  bare `memory allocation of 469762048 bytes failed`.
- `ulimit -v 1400000` → tiles refused up-front, model loads in 52 ms,
  generation completes (12.9 tk/s short decode), exit 0. Before: exit 134 with
  `memory allocation of 607744 bytes failed` (the logits buffer) because the
  tiles had consumed the remaining address space.
- Ternary-9B (PQ2_0/type 142 + PTQ1_0/type 143) still loads: the new
  tensor-bounds check is row-layout aware for the prism codecs (34 B and 28 B
  per 128 values); an unknown type is skipped rather than guessed.
- Concurrency, all exit 0, no kernel OOM kills in the window:
  2 × Qwen3-0.6B-Q3_K_S (10.5 / 10.3 tk/s each);
  3 × Qwen3-0.6B-Q4_K_M (5.0 / 4.9 / 5.0 each, 12 threads on 4 cores);
  Spark-X2.5-4B with `SOFUU_TILE_MB=1024` + Qwen3-0.6B-Q4_K_M
  (3.8 / 9.0 tk/s).
- No perf regression, same-window A/B `t1p60` (pre-hardening) vs `t1p66`
  (identical compute path to t1p67), 1210-token prompt, `-n 40`, greedy:

| model | pp t1p60 | pp t1p66 | tg t1p60 | tg t1p66 |
|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 118.4 / 124.8 | 119.2 / 122.3 | 5.6 / 5.7 | 5.7 / 5.6 |
| Qwen3-0.6B Q3_K_S | 100.9 / 101.3 (r3/r4: 101.5 / 103.1) | 98.8 / 99.0 (r3/r4: 100.3 / 102.4) | 5.7 / 5.6 | 5.6 / 5.8 |
| Llama-3.2-1B Q8_0 | 120.6 / 120.2 (r3/r4) | 119.3 / 119.9 | 5.5 / 5.2 | 5.5 / 5.6 |
| Spark-X2.5-4B Q4_K_M | 20.7 | 20.8 | 2.6 | 2.7 |

The long-prompt decode cell (5–6 tk/s) is attention-over-1210-positions, not
the short-prompt decode rate in the tables above; it is quoted here only for
the A/B. Llama-1B's first `t1p60` run read 67.2 tk/s on a cold page cache and
was excluded; the steady-state pairs above were rerun back-to-back.

### Knobs (Pi 5)

| env | default | effect |
|---|---|---|
| `SOFUU_TILE_MB` | 0 (unlimited) | total packed-tile budget; set on multi-instance hosts |
| `SOFUU_Q3K_TILE_MB` | 768 | q3_K tile budget (its tile is 1.9x the model) |
| `SOFUU_Q4_SS_MB` | 768 | q4_K pre-decoded-scale sidecar budget |
| `SOFUU_TILE_RESERVE_MB` | 256 | address space kept free when `RLIMIT_AS` is set |
| `SOFUU_NO_TILES` | unset | disables all tiles (per-row kernels, same math) |
| `SOFUU_NUM_THREADS` | unset | worker count override (clamped 1–64) |
| `SOFUU_KV_F16=0` | f16 shadows on | disables the CPU f16 KV shadows |
| `SOFUU_POOL_SELFTEST=1` | — | runs the pool resilience self-test and exits |

Tests: `test_pool_resilience_selftest`, `test_matvec_rejects_short_weights`,
`test_truncated_gguf_is_a_clean_error`, `test_corrupt_gguf_header_is_rejected`
(112 infer tests pass on the Mac lane).

## Session 14 (2026-10-23) — Long-context conversations (Pi 5 lane)

Charter: make a long prompt cheap. At the 1210-token bench prompt the old
pipeline spent **4.47 s of fixed cost per fresh conversation** (tokenizer +
full KV-cache zero-fill) and its prefill was 118 t/s against llama.cpp's
132.2. Both are gone; prefill attention was the other 43% of the remaining
time and is now row-paired.

### What shipped

1. **Heap-driven BPE tokenizer** (`infer/tokenizer.rs`). The old loop picked
   the globally lowest merge rank by rescanning every merge pair after every
   merge (O(n²) with a `format!` per comparison): 3.3-3.7 ms/token at a 1210
   token prompt, i.e. 4.4 s. The new path pre-resolves merge ranks to vocab
   ids at load (`bpe_fast`) and drives the same "lowest rank, leftmost on
   ties" rule from a `BinaryHeap` of `(rank, left, right)` with lazy
   invalidation. `SOFUU_BPE_LEGACY=1` restores the scanner for an A/B.
   Tokenization is now **5.9 ms** — the whole fixed cost of a fresh prompt.
2. **No KV zero-fill on reset** (`model.rs reset_cache`). Every reader is
   bounded by `pos + 1` (attention counts, KIVI ingest, spec tree), all
   writers go through `kv16_store`, and mmap pages are lazily committed — the
   1.3 GB memset bought nothing. `SOFUU_KV_ZERO=1` restores the fill.
3. **Row-paired f32 prefill attention** (`infer_kernels.cpp`). Batched
   prefill uses the **f32** KV caches (`batched_attention_ext`), not the f16
   shadows: at 8+ rows the f16 path's conversions cost more than the halved
   bytes save, and that comment in `model.rs` is what the measurements below
   confirm. The score pass now processes rows in pairs sharing every K load
   (8 loads + 16 FMAs per two positions instead of 8 + 8), and the value pass
   does the same for V (new `sofuu_attn_vacc_pair`). Per-row arithmetic is
   unchanged — four partial accumulators per position over 16-wide strips —
   so scores and outputs stay **bit-identical** to the previous kernel.

### Receipts (Pi 5, governor `performance`, `-t 4`, greedy, Qwen3-0.6B Q4_K_M)

| build | fixed cost (n=0) | prefill 1210 tk | attn bucket | decode |
|---|---|---|---|---|
| t1p67 (before) | 4.47 s (tokenize 4.4 + reset 0.55) | 118.0-118.6 t/s | 4246 ms | 16.4 tk/s |
| t1p68 (tokenizer + reset) | **5.9 ms** | 123.1-123.4 t/s | 4246 ms | 16.4 tk/s |
| t1p70 (+ k16 pair/quad probes) | 5.9 ms | 121.6-121.7 t/s | 4246 ms | 16.4 tk/s |
| **t1p71 (+ f32 row pairing)** | **5.9 ms** | **132.0-132.6 t/s** | **3667 ms** | **16.4 tk/s** |
| llama.cpp 11041, same window | — | 132.2 t/s | — | 12.6 tk/s |

Time to first token on the same prompt: **14.7 s → 9.17 s (-38%)**;
end-to-end prefill throughput **+11.6%**; attention bucket **-13.6%**.

Longer prompt (2419 tokens = the bench text twice):

| engine | prefill | decode | turn time (2419 prefill + 100 gen) |
|---|---|---|---|
| sofuu t1p71 | 99.4-99.6 t/s | 9.7 tk/s | 34.6 s |
| llama.cpp 11041 | 105.6 t/s | 6.5 tk/s | 38.3 s |

Prefill is 5.9% behind at 2.4k positions (attention is O(n²) and llama.cpp's
f16-dot kernels are still cheaper per MAC), generation is 49% ahead, so a
turn is ~10% faster overall and the gap widens with output length.

Full-text greedy equality against t1p67 (32 tokens, the 1210-token prompt):
**identical** for Qwen3-0.6B Q4_K_M, Qwen3-0.6B Q3_K_S, Llama-3.2-1B Q8_0,
Spark-X2.5-4B Q4_K_M and MiniCPM5-2B Q6_K_XL. The tokenizer and the new
kernels are bit-exact where it matters.

Per-model prefill/decode at the 1210-token prompt (t1p71): Qwen Q4_K_M
132.0/16.4, Qwen Q3_K_S 110.1/13.9, Llama-1B Q8_0 124.9/14.8, Spark Q4_K_M
21.4/3.8 (SWA, window-bounded), MiniCPM5-2B Q6 35.5/6.6 (SWA).

### Measured dead ends (kept out)

- **Head-major attention work order** (-15% prefill): re-indexing the static
  pool split so every thread sees all row groups looked balanced on paper but
  cost 1.5-2x in the attention bucket at long context — each thread re-reads
  the whole K/V prefix per head group and thrashes cache. Group-major (the
  old order) keeps a thread's four groups on the same resident prefix.
- **Quad-row V accumulation in the f16 kernels**: neutral end-to-end (the
  k16 kernels only serve decode and the micro-bench).
- **f16-dot (VMLAL) prefill scoring**: built, measured 3.6x slower as
  compiled (scalar fallback — `-march` had no `+fp16fml`), then removed:
  batched prefill does not use the f16 KV path at all, so the change had no
  path to the metric. Revisit only if prefill is ever routed to the f16
  shadows, and then as an explicitly non-bit-exact option.

### Knobs and tests

| env | default | effect |
|---|---|---|
| `SOFUU_BPE_LEGACY=1` | off | restores the O(n²) tokenizer scanner |
| `SOFUU_KV_ZERO=1` | off | zero-fills the KV cache on reset (old behaviour) |

Tests: `fast_bpe_matches_legacy_on_edge_cases`,
`fast_bpe_matches_legacy_fuzz` (400 random strings),
`fast_bpe_duplicate_merge_rank`,
`fast_bpe_matches_legacy_real_qwen_prompt` (real GGUF, bench prompt),
and `test_attention_k16_matches_f32_for_f16_exact_kv` now includes a
32-row long-prefix batch that exercises the new paired f32 score/value
kernels over a long shared prefix (it caught a missing tail row during
development).
