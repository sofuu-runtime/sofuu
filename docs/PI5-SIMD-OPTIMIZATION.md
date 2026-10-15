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
