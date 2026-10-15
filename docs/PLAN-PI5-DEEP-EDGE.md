# PLAN-PI5-DEEP-EDGE — going past the wall: small-model dominance on Pi 5

Status 2026-10-15: research assessment (not committed work). Written after the
prefill-plan port closed (Phase 1 neutral, Phase 2 + 5a negative, §Session 4 of
PI5-SIMD-OPTIMIZATION.md). Companion survey of the 2023-2026 paper landscape is
§3 here; every claim below is callback to our own measurements (PI5-SIMD doc)
or a cited paper.

## 1. Where the remaining physics headroom lives

Pi 5: 4× Cortex-A76 @ 2.4 GHz stock, NEON + SDOT (asimddp), no i8mm/SVE/SME,
LPDDR4X (≈17 GB/s theoretical; ~10.7-12.5 GB/s measured effective), 512 KB L2
per core, no L3.

Two regimes, two different economics:

- **Prefill = arithmetic-bound.** Measured ceilings on this chip: q8_0 SDOT
  tile 185 GMAC/s (the int8 roof), q4_K tile today ~130 GMAC/s, f32 tile
  ~120-135 projected. Attention is L1-resident at our context lengths (FA2
  measured SLOWER — §Session 4 addendum 2). Everything here is kernel math.
- **Decode = byte-bound.** ~82% of decode time is streaming weights at the
  wall; both engines sit at the same ~10.7-11.5 GB/s effective bandwidth. No
  kernel beats a fixed byte count through the controller. Decode t/s scales
  with **1 / weight-bytes-per-token** — so the only real levers are: fewer
  weight bytes (format), fewer KV bytes (quantize/evict), or fewer forward
  passes per generated token (speculative decoding).

The "unbeatable in our domain" thesis therefore has exactly three pillars, in
order of difficulty:

1. **Own a format nobody else serves well on A76** (ternary/2-bit, LUT-GEMV).
2. **Squeeze the last GEMM math for prefill** (scale-apply chain + unpack
   amortization + compiler profile-guided builds).
3. **Trade FLOPs for bytes** — spec decode + KV policy — on top of (1).

## 2. The programme (ranked; targets + cost per item)

### T0 — platform wins, days, zero model work

**Status 2026-10-15 (T0 executed):**

- **T0.1 tuned codegen — SHIPPED as opt-in build env.** Discovered the C++
  side was already A76-tuned (`zig-cc.sh` injects `-mcpu=cortex_a76`,
  build.rs adds `-march=armv8.2-a+dotprod`); the missing piece was rustc's
  own codegen. New `SOFUU_PI5_TUNED=1` (± `SOFUU_PI5_LTO=1`) env in
  `scripts/cross/build_linux_arm64.sh`: `-C target-cpu=cortex-a76` +
  thin LTO + CGU=1 (LTO must go through the cargo profile — `-C lto` in
  RUSTFLAGS errors on intermediate rlibs). Device A/B (Qwen3-0.6B,
  interleaved, 2.8 GHz box): tuned total 5.22-5.23 s vs 5.32-5.33 s baseline
  (**+1.8% total**); decode 21.3-21.4 vs 20.7 t/s (**+2.9%**, non-overlapping
  bands); prefill −1.2% (158.2-158.8 vs 159.9-160.7) — code layout cost paid
  by the decode gain. Llama-1B neutral in both directions (its decode sits
  hard on the wall). Deployment: tuned binary lives on the device as
  `sofuu-linux-arm64.tuned`; the standing `sofuu-linux-arm64` stays generic
  until peers' batteries cooperate with the swap.
- **T0.1c PGO — implementable but BLOCKED at link.** Good news: rustup's
  aarch64-unknown-linux-musl std **ships libprofiler_builtins**, so
  `-Cprofile-generate` compiles; the static-pie link then fails with
  `ld.lld: error: cannot open __llvm_profile_runtime` — zig's toolchain has
  no compiler-rt profile runtime for this target. Unblock options (either
  works): a clang-musl toolchain shipping `libclang_rt.profile`, or nightly
  `-Z build-std=std,panic_immediate_abort` with the profiler feature. The
  instrument/use env branches are already in the build script
  (`SOFUU_PI5_PGO_INSTR` / `SOFUU_PI5_PGO_USE`) — the rerun is mechanical.
- **T0.2 huge pages / TLB — CLOSED (device kernel has no THP/hugetlbfs).**
  `getconf PAGESIZE` = 16384; `/sys/kernel/mm/transparent_hugepage/` absent;
  no Huge* lines in meminfo; no config.gz. With 16K pages + A76 contiguous
  TLB entries, TLB-miss pressure on a streaming GEMV is already marginal, so
  the only route (custom kernel with THP) is bad trade for a bench box. The
  mmap layer's existing advice (MADV_SEQUENTIAL at load, header WILLNEED,
  NORMAL switch) is the correct shape for our access pattern.
- **T0.3 clocks — receipt-relevant discovery: every number this session,
  including the llama.cpp comparisons, was measured at a SUSTAINED 2.8 GHz**
  (`arm_freq=2800` in /boot/firmware/config.txt; `vcgencmd measure_clock arm`
  ≈ 2.8004 GHz under load — the "stock 2.4" in our docs refers to the SoC
  rating, not this box's config). All engine-vs-engine comparisons remain
  equal-footing (same box, same clock, governor performance, no throttle
  bits). No further overclock attempted without an explicit go — raising
  `arm_freq` requires a reboot of a machine whose boot config could brick
  remote access.

| item | what | expected | risk |
|---|---|---|---|
| T0.1 | ✅ shipped: `SOFUU_PI5_TUNED=1 SOFUU_PI5_LTO=1` cross-build (rustc target-cpu + thin LTO + CGU=1; C++ side was already tuned) | measured +1.8% Qwen total, +2.9% Qwen decode | low (opt-in env) |
| T0.1c | PGO infra in script; blocked at static-musl link (missing compiler-rt profile runtime) | +3-8% once toolchain unblocks | medium |
| T0.2 | closed: no THP/hugetlbfs in the device kernel; 16K pages + contiguous TLB make it marginal | — | — |
| T0.3 | documented: sustained 2.8 GHz is the standing box policy (all receipts equal-footing) | fairness framing held | — |

### T1 — the prefill kill shot (weeks)

- **T1.1 q4_K scale-apply chain** — the one remaining new-maths lever. Fold
  the per-subblock scale application (vmulq_s32/vmovl/vcvtq + bsums bias
  vmlals, ~130 vs 185 GMAC/s gap vs q8_0) into the accumulator fold — same
  two-sided per-band fold the parked Phase-2 SDOT tile proved. Target ~150
  GMAC/s → **Qwen prefill 158 → ~180 t/s (past llama.cpp's 177), Spark ≥ 23
  (past 22.6)**. Gates: vector-rel-RMS + argmax, then battery.
- **T1.2 Arm-paper techniques (arXiv 2501.00032)** — their 3-3.2× prefill /
  2× decode vs llama.cpp rests on amortizing weight-unpack across output rows
  + interleaved group layout + non-uniform codebooks. Our panels already share
  unpacked rows across tokens; the delta to steal is the interleaved group
  weight layout at repack time and their dequant-path shape. Fold into T1.1
  as one GEMM attack so it lands once, with one battery.

### T2 — the decode moat: fewer bytes per token (the "no one can follow without training their own model" pillar)

- **T2.1 BitNet b1.58 2B4T support (start here — a real model exists).**
  microsoft/BitNet-b1.58-2B4T-gguf is a native-ternary 2B trained on 4T
  tokens, ~0.4 GB, reported to match full-precision 2B-class models. Serve it
  with our own i2_s / TL1-style LUT GEMV kernels (vqtbl-based, T-MAC
  mechanism): decode wall math says ~26 t/s max, ~20-24 realistic (T-MAC
  measured 11 t/s on Pi 5 for the 3B: 78% of wall). Prefill also LUT-able.
  llama.cpp mainline cannot do ternary LUT well — this is the format moat.
- **T2.2 Own QAT ternary 0.6B (the endgame).** ParetoQ (NeurIPS'25, Meta):
  ternary 600M beats prior SOTA ternary 3B — ternary is the size-quality
  sweet spot, and Spectra shows PTQ below 3 bits collapses (QAT is the route).
  Either train from scratch on our tokenizer/dataset line, or distill-convert
  a 0.5B FP checkpoint (BitNet Distillation / "An Extra RMSNorm" recipes). A
  ~0.15-0.2 GB model ⇒ wall says **40-55 t/s decode**, ~2.5× our current 0.6B
  number. Needs a training decision from the developer (ties into the emb-m
  tokenizer/data infrastructure).
- **T2.3 KV bytes at long context.** Stage 1: q8_0/q5_0 KV (llama.cpp-parity
  class, one-day flag). Stage 2: SnapKV-style prefill-time eviction (all cost
  paid during prefill where we have compute headroom) + DuoAttention-lite
  head split. Stage 3 (own-model): MHA2MLA/TransMLA-style low-rank KV
  retrofit (−90% KV bytes, proven on 2-7B). Caution to carry: EdgeLLM-Nano
  measured KV-compression backends LESS energy-efficient per token at tiny
  scale — measure quality the LCDF way before adopting.
- **T2.4 Speculative decode (peer lane — trees/KV are theirs; coordinate).**
  With T2.2's model: LayerSkip-style self-draft (draft = first ~4 layers, no
  second model) or a Kangaroo-style adapter trained in. Port EAGLE-2's
  confidence-as-acceptance calibration and Sequoia's hardware-aware tree
  sizing onto our lookahead/tree machinery. Realistic ceiling on a 4-core
  consumer CPU measured by the 2026 anatomy study: **~1.5-1.8× wall-clock** —
  on 45-55 t/s base that is **60-80 t/s effective decode on a 0.6B-class
  model**, the honest meaning of "unreal".
- **T2.5 LM-head vector index (sketch-level).** Replace the tied-vocab output
  GEMV with HNSW search — 2026 paper claims up to 82% CPU decode gain;
  prototype on 0.6B where the head is a large share of streamed bytes.

### Target scoreboard ("no one can beat this on this class of silicon")

| workload | today | target | mechanism |
|---|---|---|---|
| Qwen3-0.6B q4_K prefill | 158 (0.89× llama.cpp) | **≥180 (1.02×)** | T1.1 + T0.1 |
| Qwen3-0.6B decode | 20.6 (wall parity) | 20.6 (wall — unassailable at q4 bytes) | nothing to do: wall |
| Llama-1B Q8_0 prefill | 133 (2.28×) | ≥133 held | regression gates |
| Spark-4B q4 prefill | 22.2 (0.98×) | ≥23 (1.02×) | T1.1 |
| BitNet-2B decode | n/a | **20-24 t/s** (T-MAC: 11 on the 3B) | T2.1 |
| Sofuu-Ternary-0.6B decode | n/a | **45-55 t/s; 60-80 with spec** | T2.2 + T2.4 |
| 16k-context decode | attn-dominant | 2-4× | T2.3 |
| load | fastest | held | regression gates |

## 3. Paper survey (2023-2026, verified against arXiv/GitHub this session)

Legend: ✱ = active try next; 🧭 = competitor to track; 📚 = context/foundation.
Full details incl. claimed numbers in the session chat transcript; this is the
durable map.

### 3.1 CPU/ARM inference engines (competitors + techniques)
- ✱🧭 **Arm: Highly Optimized Kernels and Fine-Grained Codebooks for LLM
  Inference on Arm CPUs** — arXiv 2501.00032 (Arm, Dec 2024). 3-3.2× prefill,
  2× decode vs llama.cpp on 4-bit; row-batched unpack amortization +
  interleaved group layout + non-uniform codebooks. Our T1.2 source.
- ✱🧭 **MNN-LLM** — arXiv 2506.10443 / MMAsia'24 (Alibaba). 8.6× prefill /
  2.3× decode vs llama.cpp on phones (leans on i8mm we lack); steal: per-ISA
  tile shapes, DRAM-flash KV spillover, KV keys int4 + values fp8.
- 🧭 **T-MAC** — EuroSys 2025, arXiv 2407.00088 (Microsoft). LUT mpGEMM, ≤4×
  vs llama.cpp; **BitNet-b1.58-3B: 11 tok/s on Raspberry Pi 5** — the
  published number to beat in our exact domain. github.com/microsoft/T-MAC.
- 🧭 **bitnet.cpp / "1-bit AI Infra" + "Bitnet.cpp: Efficient Edge Inference
  for Ternary LLMs"** — arXiv 2410.16144, 2502.11880. I2_S / TL1 / TL2 / ELUT
  kernels; ARM 1.37-5.07×. github.com/microsoft/BitNet.
- 🧭 **Intel ultra-low-bit CPU microkernels** — arXiv 2508.06753 (Aug 2025):
  2-bit close-to-roofline CPU kernels, 2.2× over bitnet.cpp. Watch for code.
- 📚 **KleidiAI** (Arm) microkernels; llama.cpp merged a backend Feb 2025.
  Biggest wins are i8mm/SME-class, not A76 — monitor PR outcomes only.
- 📚 **PowerInfer / PowerInfer-2** — arXiv 2312.12456 (SOSP'24) / 2406.06282.
  Activation-sparsity scheduling; the transferable piece is sparse-FFN
  decode (see Turbo Sparse 2406.05955, SmallThinker 2507.20984 for
  natively-sparse CPU models — an own-model option).
- 📚 **MLC-LLM/TVM AOT** — AOT-compiled fused dequant+GEMV as an alternative
  kernel-generation path; no standalone paper (WebLLM 2412.15803 covers the
  pipeline). **nncase** (arXiv 2512.21571, Dec 2025): e-graph LLM compiler
  matching hand-tuned llama.cpp on CPU.
- ✱ **Large pages/TLB for decode** — nothing published; our T0.2.

### 3.2 Attention, KV, long context
- 📚 **FlashAttention-2** (2307.08691) / FA3 (2407.08608): wins only when S²
  or KV overflows cache — our L1-resident measurement agrees; revisit only
  >2k contexts. oneDNN's own SDPA path uses the same size threshold.
- 🧭 **KIVI** (ICML'24, 2402.02750): 2-bit KV, GPU-shaped (per-channel K needs
  transposes); no CPU port exists; llama.cpp Q8_0/Q5_0 KV is the pragmatic
  CPU version — our T2.3 stage 1.
- ✱ **SnapKV** (NeurIPS'24, 2404.14469): prefill-time KV selection — pays in
  prefill, saves decode reads. **H2O** (2306.14048), **TOVA** (2401.06104),
  **StreamingLLM sinks** (2309.17453 — we already run sink semantics),
  **PyramidKV** (2406.02069), **DuoAttention** (2410.10819): the eviction
  family; DuoAttention's head split is the right granularity for our GQA
  small models.
- ✱ **MHA2MLA** (ACL'25, 2502.14837) / **TransMLA** (2502.07864): MHA/GQA →
  low-rank KV retrofit, −90%+ KV bytes; own-model route in T2.3.
- 📚 **PagedAttention** (SOSP'23): single-user CPU takeaway is allocate-as-
  you-decode + block indirection as a substrate for tiered KV, nothing more.
- **CacheGen** (SIGCOMM'24, 2310.07240), **CachedAttention** (ATC'24),
  **TierKV** (2609.21172, Sep 2026): KV-to-flash tiers — matches our
  DRAM-flash ideas from MNN-LLM; only for models near RAM size.
- ⚠️ **EdgeLLM-Nano** (2610.14475, Oct 2026): KV-compression backends measured
  less energy-efficient per token at tiny scale — the "measure first" flag.

### 3.3 Quantization + LUT GEMV (the format pillar)
- ✱ **T-MAC + bitnet.cpp TL/I2_S** (above): current LUT state of the art.
  **LUT-GEMM** (ICLR'24, 2206.09557, NAVER) is the prior art.
- 📚 **BitNet family**: 2310.11453 (BitLinear), 2402.17764 (b1.58), 2411.04965
  (a4.8 — 3-bit KV + ~55% activated params), **2504.12285 (b1.58 2B4T** — the
  open-model vehicle for T2.1).
- ✱ **ParetoQ** (NeurIPS'25, 2502.02631, Meta): unified 1/1.58/2/3/4-bit QAT;
  ternary 600M > prior SOTA ternary 3B; ternary/2/3-bit beats 4-bit on the
  Pareto curve — the citation behind T2.2.
- 📚 **Spectra** (2407.12327): TriLMs 99M-3.9B; PTQ collapses below 3 bits,
  QAT is the route — bounds the T2.2 design space.
- **Distill-convert recipes**: BitNet Distillation (2510.13998), "An Extra
  RMSNorm is All You Need" (2505.08823) — cheap path from an existing 0.5B FP
  checkpoint to ternary.
- **2-3-bit PTQ quality keepers** (for GGUF-fleet models, not our moat):
  AQLM (2401.06118, codebook = LUT, has CPU kernels), QuIP# (2402.04396, E8
  lattice + Hadamard — butterflies are free on NEON), SqueezeLLM (2306.07629,
  dense-codebook + sparse outliers, fully LUT), spQR (2306.03078); Known:
  GPTQ/AWQ/OmniQuant/SmoothQuant.
- ⚠️ **OneBit** (2402.11295) / BiLLM (2402.04291) / PB-LLM: 1-bit is where
  quality dies on small models — ternary, not binary.

### 3.4 Speculative / parallel decoding
- 📚 **Speculative sampling** (Leviathan/Chen 2023), **Medusa** (2401.10774),
  **Hydra** (2402.05109), **EAGLE-1/2/3** (2401.15077, 2406.16858 — tau
  4.64-4.68 on 7B, 3.05-4.26×; 2503.01840), **Lookahead** (2402.02057 —
  ours), **SpecInfer** (2305.09781), **Sequoia** (2402.12374 —
  hardware-aware tree sizing: port).
- ✱ **Draft & Verify / self-speculative** (ACL'24, 2309.08168),
  **LayerSkip** (2404.16710 — draft = early layers, most RAM-friendly),
  **CLaSp** (2505.24196, training-free layer-skip drafter), **Kangaroo**
  (2404.18911, 67M adapter), **CLLMs** (2403.00835), **PEARL** (2408.11850).
- ⚠️ **"Lossless but Not Free"** (2607.17283, 2026): measured spec decoding on
  consumer CPUs — best 1.61× wall, acceptance drops from ~70% to ~38% at
  optimal K, 3 of 5 configs SLOWER. Our trees must batch-verify or it nets
  negative. [NEW 2026 successors: BitNest (2610.02800 — draft from a
  low-precision copy nested in the target weights; natural pairing with our
  parked int8 path), SEED, UBTree, D-Loop — none target 4-core NEON CPUs.]
- 🧭 **Edge spec**: MemSpec (LCTES'26), S2-MoE (llama.cpp MCP). **No paper
  measures spec decoding on a Pi-class board — our receipts would be
  first-in-literature.**

### 3.5 What the survey says about our position
- Nobody has published A76-dotprod-specific q4_K kernels, a Pi-5 spec-decode
  receipt, or the CPU crossover point for FA-vs-two-pass attention (our 1k
  measurement is currently literature-unique). The moat = receipts + gates +
  battery culture, extended by format ownership (T2.2) — competitors can port
  the same papers, but our battery turns "claims" into "numbers" faster.
- Kernel papers that assume i8mm/SME (Arm/Kleidi/MNN-LLM's headline) do NOT
  transfer to A76 at face value; the SDOT-class subset is ours to take.

## 4. Edge cases / honesty

- The decode wall (~10.7-12.5 GB/s effective) is shared by every engine on
  this SoC, including all competitors. "Unreal" decode can only mean fewer
  bytes/token (format) or fewer forwards/token (spec). Anything else is
  measurement noise or unfair clocks.
- Ternary decode numbers assume we serve a ternary model: without T2.1/T2.2
  there is no byte advantage over q4 — the wall stands.
- Own-model training (T2.2) is a developer decision: it competes with the
  emb-m/GigaToken workstream for resources and needs a quality gate vs
  Qwen3-0.6B Q4_K_M at task level, not just perplexity.
