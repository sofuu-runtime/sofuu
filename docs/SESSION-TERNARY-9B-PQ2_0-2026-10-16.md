# Ternary-Bonsai-2-9B (qwen35 / PQ2_0 + MTP) + MiniCPM5-Sharp Support Session — 2026-10-16

## Scope

Add CPU-only inference support for the two new checkpoints plus disposition of
the MiniCPM5-Sharp file:

- `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` (2.68 GB) — qwen35 arch, GGML
  **type 142 PQ2_0** mainline weights, `prism.hadamard.block_size = 4096`,
  `qwen35.nextn_predict_layers = 1` (MTP draft packed as blk.32), 32 main
  layers (24 GDN + 8 attention, interval 4).
- `Ternary-Bonsai-2-9B-PTQ1_0.gguf` (2.00 GB) — same model, type-143 trit
  codec, used as the codec-difference control.
- `Sharp-MiniCPM5-2B-Q6_K_XL.gguf` (2.20 GB) — plain `llama` arch, standard
  Q6_K (189) + Q8_0 (107) + F32 (85) tensors; no custom quant, no exotic keys
  (key_length 128, kv 2, ctx 131072, rope base 5e6 full-width 128).

All targets exclude GPU paths; the sharp-MiniCPM5 file turned out to need
**zero code changes** (see verdict below).

## Model facts established

- **PQ2_0 = GGML type 142, PrismML's 2-bit packing**: 34 B per 128 values =
  `ggml_half d FIRST` + `uint8 qs[32]`; value j at byte j/4, bits
  (j%4)·2 lowest-first; dequant `(code − 1) · d` — the {−1, 0, +1, +2}·d
  lattice (2.125 bpw). Same family as PTQ1_0 (143, 28 B/128 trits).
  Format taken from the public `PrismML-Eng/llama.cpp` branch `prism`
  (`ggml-common.h` struct + `ggml-quants.c` codec) and pinned by an
  independent Python reader: sha256 of the first 128 dequantized f32 values
  of `token_embd` row 0 =
  `79ef9d8b5a06cd43e648b72e64f5fd5335b46b733cd331a668a0ede1ceb437b8`
  (block-0 bytes `bb2205960aaa6054…`, d = 0.01314544677734375). Engine test
  `qwen35::tests::pq2_0_ternary9b_token_embd_decode_reference` pins byte
  view + scale + sha; `pq2_0_dequant_hand_block` pins the codec arithmetic.
- **prism.hadamard v1, block 4096**: same normalized Sylvester/Wash-H
  construction as the 27B (version 1, `sign_mode = explicit`,
  `sign_values` = two sign rows, widths {4096, 12288}). `wht_block` was
  already size-generic; `HAD_BLOCK` became a per-model field (whitelist
  {1024, 4096}), with sign-width divisibility validated at load.
- **`prism.hadamard.gdn_v_grouped = true` on BOTH the 27B and the 9B** (my
  first 27B KV probe was broken and said only the 9B had it). Semantics per
  the fork's graph code: for `ssm_out`, the v activation arrives
  [hd, nk, rep]-tiled and the runtime regroups [hd, nk, rep] → [hd, rep, nk]
  *before* signs+WHT — i.e. exactly our existing `gdn_perm`, repositioned
  under the ssm_out transform for flagged checkpoints. **NOT "skip the
  perm"**: an identity interpretation was A/B-tested and makes even the 27B
  incoherent ("()/()/()…"), while the perm path keeps both models sane.
  `SOFUU_GDN_V_RAW=1` remains as the A/B kill knob.
- **The 27B regression that exposed this**: `gdn_v_grouped` gating defaulted
  the 27B to identity and produced ")/()/()…" — caught by re-running the
  capital-of-France control after the flag landed.
- **MTP/nextn**: `blk.33`(absent) — the draft layer IS `blk.32` holding
  `nextn.eh_proj/enorm/hnorm/shared_head_norm` + an attention set + ffn, all
  Q8_0. Main model = `block_count − nextn_predict_layers` = 32 layers; the
  draft tensors are skipped everywhere for v1 (self-spec decode is future
  work — SPECULATIVE.md-style, MTP head unused).
- **ssm_alpha/ssm_beta dtype**: 27B ships BF16, the 9B ships F32 — loader
  accepts both.
- The 8 Q8_0 tensors in the PQ2_0 file are all blk.32 (draft) tensors;
  every main-layer weight is type 142. There are no `*_s` per-channel scale
  tensors in either file.
- **Embedding decode was PTQ1_0-hardwired** (`ptq1_0_dequant_row`) in both
  embed paths — decoding PQ2_0 bytes as trit blocks yields nonsense f16
  scales and a residual stream ~340× too hot (h_rms 393 vs 0.25). Both
  embed sites now dispatch on the tensor type
  (`dequant_row_to_f32`).

## What was done (all in `crates/sofuu-core/src/infer/qwen35.rs`)

1. `GGML_TYPE_PQ2_0 = 142`, block constants, `dequant_pq2_i8`
   (slab contract parity with `dequant_trits_i8`), codec-dispatched
   `dequant_row_i8` in every GEMV/GEMM/slab dequant site, file-direct
   `packed_row_dot_t142` + `packed_row_dot` dispatch.
2. `raw_mat_of` accepts either prism codec where PTQ1_0 was expected and
   carries the actual type (`RawMat.ty`); row_bytes switch.
3. Hadamard: `HAD_BLOCKS {1024, 4096}` whitelist, per-model `had_block`
   threaded through `wht_blocks`/`hadamard_prep{,_in_place}` (12 call
   sites), sign-width divisibility check, `prism.hadamard.version ≤ 1`.
4. MTP draft-layer skip (`n_main_layers = block_count −
   nextn_predict_layers`), config.n_layers set to the main count.
5. `bf16_meta_mat` accepts F32 or BF16 (ssm_alpha/ssm_beta).
6. `gdn_perm` derives rep from the width (27B 3, 9B 2 — rep = dt_rank/nk);
   regroup always applied, `SOFUU_GDN_V_RAW=1` for A/B.
7. Embedding decode codec dispatch (`dequant_row_to_f32`) at both embed
   sites. Env-gated per-layer `SOFUU_DEBUG_LAYERS` norm trace (this is what
   localized the embedding bug: h_rms 393 at layer 0 = impossible for ±0.013
   embeddings).

## Verification

- Quant.rs/qwen35 suite: **598 passed / 0 failed** (two new PQ2_0 tests).
- Mac CPU, temp 0, "The capital of France is":
  - 9B PQ2_0-MTP: " Paris. / The capital of France is not London." ✅
  - 9B PTQ1_0 control: coherent ✅ (3.0 tk/s decode)
  - 27B control: " Paris. / The capital of Germany is Berlin." ✅ (both flag
    orders A/B'd; perm is load-bearing)
- Pi 5 (t1p6, 4 threads, 2.8 GHz): 9B PQ2_0 coherent " Paris. / The capital
  of France is Paris." — v1 dequant path is memory/decode bound:
  prefill 0.3 tk/s, decode 0.1 tk/s (packed i8 slab like the 27B's is
  impossible at 8.9 B params × i8; the speed program = T2: fast 2-bit band
  dots + mmap-friendly packing. Reference fork runs the same file at
  15.1 t/s generation on Apple silicon — ~10-17× over our v1 path is on the
  table by its own architecture).
- **MiniCPM5-Sharp was never broken.** The capital-of-France probe returns a
  Python `check_python_version` snippet — quirky, but this "Sharp" 2B is a
  reasoning-tuned checkpoint (the reference fork prints "[Start thinking]"
  for it) with a code-heavy base. Story prompt: "Once upon a time, there was
  a little girl who **lived in a small house. She had a big heart and a big
  smile. She loved to help others. One day, she decided to help a little
  boy**…" — coherent everywhere.
- MiniCPM5-Sharp Pi receipts (t1p6): load 3.9 s, prefill **17.6 tk/s**,
  decode **4.8 tk/s**. llama.cpp 11041 same file/prompt: 33.3 / 5.3 tk/s —
  **decode parity (DRAM wall)**; prefill 1.9× behind for the Q6_K/Q8_0
  llama paths → first row of the next optimization table.

## Formats/spec sources

- PrismML-Eng/llama.cpp branch `prism` (ggml-common.h `block_pq2_0`,
  ggml-quants.c codec, `src/models/qwen35.cpp` + `ggml_` graph code for
  hadamard/gdn_v/nextn handling) — the format reference; MODEL-FORMATS.md
  corroborates type ids 142/143 and group-128 membership.
- No local engine changes copied from unlicensed sources beyond interface
  knowledge; the codec arithmetic is two lines of bit math.
