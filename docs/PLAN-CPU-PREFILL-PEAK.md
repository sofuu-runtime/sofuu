# PLAN — CPU-only prefill to its peak (M2 Pro floor study)

> Status: **DRAFT — awaiting developer approval.** Nothing below is started.
> Author lane: `crates/sofuu-core/src/infer/lfm2.rs`, `qwen35.rs`, `gguf.rs` (the
> files this workstream already owns). Peer-owned files stay untouched
> (`quant.rs`, `model.rs`, `infer_kernels.cpp`, `main.rs`, `spark.rs`,
> `chat.rs`, `main.rs` in CLI) — see execution rules.

## 0. Ground truth (measured 2026-10-14, committed `845f3e35b`)

| Quantity | Value | Source |
|---|---|---|
| Chip | M2 Pro, 6 P-cores ~3.5 GHz + 4 E-cores, 200 GB/s, 16 GB | `sysctl` |
| NEON f32 peak (6 P-cores) | ~670 GFLOP/s | 32 FLOP/cyc/core |
| tile4x4 kernel, 1 thread | 85 GFLOP/s | `tile4x4_gflops_single_thread` |
| tile4x4 kernel, 6 threads | **493 GFLOP/s (5.78×)** | `tile4x4_gflops_six_threads` |
| Whole prefill (liquid-1.3B, 200 tok) | 47.8 tk/s = **124 GFLOP/s** | today's serve path |
| Absolute prefill ceiling (f32 NEON) | **~258 tk/s** (2.6 GFLOP/tok) | arithmetic |
| Ternary 27B prefill today | 1.65 tk/s (27 GMAC/tok) | calm serve, `71469e71e` era |

Key diagnosis: the kernel is at 74% of the f32 peak; the *end-to-end* prefill
is at 19%. **The remaining 55% is glue** (serial norm/attention/SwiGLU
segments, 130 pool dispatches per prompt, dequant + residual passes), not
math. So the plan is ordered by cost-of-change per token-per-second gained.

## Phase 1 — Glue fusion (safe, pure engineering)  → target 90–150 tk/s

- [ ] Fuse `batched_rmsnorm` into the GEMM prologue: dequant row → w·(x·rms)
      per panel instead of a separate pass over `xn` (saves a 1.6 MB r/w per
      layer call and one pool dispatch).
- [ ] Merge `out_proj + residual` and `down + residual`: fold `x[t] += row`
      into the tile stores (removes a full T×dim pass per layer).
- [ ] Panel size sweep (16 → 32/64) + back-to-back matvec batching
      (gate+up in one pass: one x sweep, two accumulators) — fewer dispatches,
      better L2 reuse of the x rows.
- [ ] Parallelize the remaining serial per-token attention segment only if
      profiling still shows it >10% at T=200 (heads are independent).
- Acceptance: suite green; batched-vs-per-token logits gates unchanged;
      measured ≥ 90 tk/s on liquid-1.3B, 200-tok, calm machine.
- Risk: low. All inside `lfm2.rs`. No flag needed (bit-parity not claimed —
      RMS gates stay).

#> **DEFAULTS UPDATE 2026-10-15:** the ladders are the standing behavior, not experiments —
> AMX-on-macOS default-on (exact-f32, `SOFUU_PREFILL_BLAS=0` to opt out), 27B direct tile
> default-on (bit-identical, `SOFUU_T143_DIRECT=0` to opt out), int8 two-sided stays
> opt-in (`SOFUU_PRE8=1`) for non-AMX aarch64. Same contract for every backend: batched
> prefill + dequant-once + L2-fit slab + pinned gates.

# Phase 2 — int8 activations in the float-weight GEMM  → target 200–300 tk/s

> **DONE 2026-10-15 (ca696851d):** shipped as two-sided 32-group quant
> (`SOFUU_PRE8=1`) + per-load `Pre8Slab` weight cache. Warm liquid-1.3B:
> 79.0 tk/s (f32 68.1 same window, +16%). Kernel bit-exact vs i64 ref; RMS
> 0.05 bf16 / 0.049 mlx4, greedy identical. 128-band absmax (RMS 0.108) broke
> greedy — 32-group granularity is the correctness floor. Scalar quant-pass
> over 2.6B weight elements cost 15% e2e until vectorized. Perf target not
> met (200–300 assumed sdot 10×; measured sdot reality is ~24%/core over f32) —
> AMX (Phase 3) beat it instead.

- [x] Quantize activations per 128-band once per layer call (reuse the
      `quantize_acts`/ActQ machinery from `qwen35.rs`), then sdot int8 tile
      variant of `gemm_tile4x4_neon` (weights still dequantized from bf16/MLX4
      into int8 lanes with per-group scales).
- [x] Two-sided fold per band (w_scale · a_scale · iacc), mirroring
      `trit_row_dot_q` — same pattern already proven bit-stable on the ternary
      path.
- [x] Threshold test budget: rel-RMS < 1e-2 vs f32 path on real model
      (the ternary path's existing gate), greedy top-1 equality over 200-tok
      prompts; ill-conditioned bands (absmax 0) keep the existing fallback.
- Acceptance: ≥ 1.8× over Phase 1 on the same machine; greedy continuations
      identical over 3 prompts × 64 tokens on liquid + liquid-350.
- Risk: medium (numerical). Flag `SOFUU_PRE8=1` until receipts, then default-on.
- Note: the k-quant GGUF int8-GEMM already has its own workstream (peer's
      `SOFUU_BENCH_GEMM` hook); this phase is strictly the bf16/MLX4 float
      fleet in `lfm2.rs`.

## Phase 3 — AMX via Accelerate (CPU-die coprocessor, not the GPU)  → target 200–350 tk/s

> **DONE 2026-10-15 (8f59dc6e0):** warm liquid-1.3B **104.5 tk/s** (int8 79.0,
> f32 68.1 same window; 4.23× per-token), gemm stage 2937 → 1607 ms. Binary
> byte-identical (5,097,392 B) — dylib link. Whole-tensor dequant on the ffn
> pool, one row-major sgemm per tensor on Accelerate's pool (no nesting).
> Exact-f32 gates pass untouched. 200–350 target assumed bigger tiles reach
> peak AMX; real ceiling on this workload is the dequant+sgemm split, not AMX
> throughput. Headroom left: ~104 ms/call w32-slab alloc churn.

- [x] Link `Accelerate.framework` (additive `#[link(name = "Accelerate")]`
      extern block in `lfm2.rs`; no Cargo dependency changes needed).
- [x] Opt-in env `SOFUU_PREFILL_BLAS=1`: per layer call, dequant the bf16/mlx4
      weight once into an f32 buffer (thread-local scratch), run
      `cblas_sgemm` for the [T×cols]·[cols×rows] product, then fuse residuals.
      Break-even at T ≈ 64–128 (dequant amortization); keep the NEON tile path
      below that and as fallback (also covers the 5 MB binary-gate check).
- [x] Cross-check Apple BLAS numeric contract (row-major details) with one
      parity test at T=133 (ragged).
- Acceptance: ≥ 2× over Phase 2 at T ≥ 128; binary size gate still ≤ 5 MB;
      suite green with flag off by default.
- Risk: medium (build/link; memory churn from the f32 weight copy — measure
      with the residency interaction; worst case keep flag opt-in permanently).

## Phase 4 — T-MAC-style LUT prefill (and decode) for the ternary 27B  → 1.65 → 20–40 tk/s

- [ ] Study + port the T-MAC idea (arXiv 2407.00088,
      github.com/microsoft/T-MAC): group 4 ternary weight values per LUT,
      precompute int8 partial sums per activation group; accumulate via
      `vqtbl4q_u8`-family table lookups + shift/or-aggregation across bit
      planes (W1.58 ⇒ 3 planes).
- [ ] Kernel sketch: `lut_bgemm_t143(blocks, aq)` producing per-128-band i32
      sums — output must remain **bit-identical** to the existing packed
      `sdot` path (i32 band sums are order-independent; extend
      `packed_row_dot_bitmatches_dequant_sdot_path`), which makes LUT a drop-in
      for both `gemv_t143` (decode) and the prefill row-slabs.
- [ ] Resident interaction: with `SOFUU_RESIDENT_WEIGHTS=1` the LUT rebuild
      cost is paid once per weight per process (27B prefill LUT table ≈
      weights/4 × 81 B — measure; if too big, keep LUTs per-band-scratch
      in registers per call instead).
- [ ] First milestone: single-block unit test (bit-exact), then 200-tok
      prefill A/B on the 27B vs today's `gemm_rows_t143_sdot`; MLPerf-style
      receipts in TASKS.md.
- Acceptance: ≥ 8× on 27B prefill ≥ 128-tok prompts; decode parity retained
      (greedy chains unchanged); suite green.
- Risk: high (research-scale kernel; LUT memory sizing on 16 GB). This phase
      is standalone — it does not gate Phases 1–3.

## Phase 5 (conditional) — long-context prefill + serving shape

- [ ] FlashAttention-2-style tiled CPU attention once attention > 15% of
      prefill (T ≥ ~1000; today 6% at T=200).
- [ ] Windowed prefill with sink tokens for max_seq_len ≥ 4k configs.
- [ ] Chunked prefill scheduling (Sarathi-Serve, arXiv 2403.02310) only if
      the engine starts serving concurrent requests.
- [ ] KV-cache int8 (KIVI, arXiv 2402.02750) extends these wins to the cache
      side (hooks already exist in `model.rs` — coordinate with the peer that
      owns them).

## Execution rules

1. One phase at a time; measure before/after on a calm machine
   (`uptime < 4`), serve + `bench_sofuu.py` for end-to-end, `#[ignore]`
   probes for kernel-level numbers; record `wall + GFLOP/s` in TASKS.md.
2. Quality gates ride with every phase: full `cargo test -p sofuu-core
   --release --lib` green, batch-vs-per-token RMS + greedy-continuation
   tests, and the 5 MB binary gate.
3. Behavior-changing paths start flag-gated (`SOFUU_PRE8`, `SOFUU_PREFILL_BLAS`),
   default off, promoted to default only after receipts.
4. `git status` before each commit; narrow staging; straight to main; no
   OpenQodex (developer waived 2026-10-14). Peer-active files are read-only —
   if a build breaks inside them, wait, never patch.
5. Stop points: end of each phase is a natural review stop with measured
   receipts before touching the next.

## Papers / references

- **T-MAC** — LUT-based mpGEMM for low-bit LLMs on CPU;
  arXiv 2407.00088 (EuroSys'25), github.com/microsoft/T-MAC.
  W2 7B prefill 50.1/94.4 tok/s @4/8 threads; AMX only comparable at 2-bit
  multi-batch → LUT is the ternary endgame on CPU.
- **Sarathi-Serve** — chunked prefill scheduling; arXiv 2403.02310 (OSDI'24).
- **LLM in a flash** — Apple; flash-resident weight loading; arXiv 2312.11514
  (mostly covered by our residency flag; relevant under RAM pressure).
- **FlashAttention-2** — arXiv 2307.08691 (tiling scheme for Phase 5).
- **KIVI** — 2-bit KV cache quantization; arXiv 2402.02750 (Phase 5).
- **SnapKV** — attention-observation cache pruning; arXiv 2404.14469
  (already shipped in-engine post-prefill).
- **BitNet b1.58** — ternary-weight LLMs; arXiv 2402.17764 (why the 27B
  tolerates low-precision math).
- **Accelerate/BLAS + AMX** — Apple platform docs; AMX is the CPU die's matrix
  coprocessor, distinct from the GPU (in-constraint under "CPU only").
