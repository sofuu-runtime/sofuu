# SofuuBlock v1 — Architecture Specification

This is the exact specification the code in `../sofuu_arch/` implements.
Code and document must agree; `tests/test_model.py` enforces the parameter
tables mechanically. Status: **implemented, tested, unmeasured** — every
quality claim below is a hypothesis until the Phase 3 ablations run.

v1 changes over v0: sparse MoE feed-forwards (§3d), QK-normalization (§3e),
stochastic depth (§3f), fused-SDPA attention kernel (§3g, same math).

## 1. Notation

- `V = 32768` vocabulary size, `n` sequence length (train: 256).
- `d = 256` hidden dim, `h = 4` heads, `dh = 64` head dim, `L` layers.
- `f` FFN inner dim (768 for 20M/15M configs, 512 for 10M).
- `B = 32` relative-position buckets, `D = 256` max modeled distance.
- `x ∈ R[n, d]` block input; all linears are **bias-free**.

## 2. Input embeddings

Two styles, chosen per budget (ablation (f) decides the long-term default):

- **direct:** `E ∈ R[V, d]`, `x_i = E[ids_i]`.
- **factorized:** `Ef ∈ R[V, e]`, `e = 128`, projection `P ∈ R[e, d]` (no
  bias), `x_i = P · Ef[ids_i]`.

Then `x = Dropout(RMSNorm(x))`. No token-type embeddings, no absolute
position embeddings — position enters only through §4.

## 3. SofuuBlock

Pre-norm residual block, applied `L` times:

```
h = x + Attn(RMSNorm(x), mask, is_global, window)
y = h + FFN(RMSNorm(h))
```

### 3a. Disentangled attention

Per layer, projections `Wq, Wk, Wv, Wo ∈ R[d, d]` and a relative table
`Erel ∈ R[B, d]` (see §4). With `Qc = xWq`, `Kc = xWk`, `Vv = xWv`, and
`R[i, j] = Erel[b(i − j)]` reshaped per head:

```
A[i, j] = Qc[i]·Kc[j] / sqrt(d)   (content → content)
        + Qc[i]·R[i, j]            (content → position)
        + R[i, j]·Kc[j]            (position → content)
```

`softmax(A + mask) · Vv`, then `Wo`. Per-layer attention cost:
`4d² + B·d` params (262,144 + 8,192 at d=256).

### 3b. Local / global rhythm

Layer `i` (0-indexed) is global iff `(i + 1) % 3 == 0`; all other layers
are sliding-window local with radius 128 each side. Non-attended positions
(mask padding or out-of-window keys) receive `−inf` before softmax, so
their weights are exactly zero. At the 256-token training length the
window covers everything; the rhythm pays off at the 512-length
experiments and costs nothing now.

### 3c. Gated-SiLU FFN

```
FFN(x) = Wdown( silu(Wgate(x)) ⊙ Wup(x) ),   Wgate, Wup ∈ R[d, f], Wdown ∈ R[f, d]
```

Cost `3·d·f` (589,824 at f=768; 393,216 at f=512). Dropout applied to the
block output.

### 3d. Sparse MoE feed-forward (alternate layers)

Layers with odd index use `SparseMoEFFN` instead of the dense FFN: `E = 8`
experts, each a gated-SiLU FFN with inner dim `f/E` (96 for the 768 configs,
64 for the 10M config), plus a bias-free linear router `R[d, E]`. Per token:
`e* = argmax(softmax(xR))`, `y = prob[e*] · expert[e*](x)`.

The sizing is the whole point: **8 experts of `f/8` cost exactly what one
dense `f` FFN costs in total params**, so MoE layers add only the router
(`d·E` = 2,048 params). Active compute per MoE layer is ~1/8 of dense.
Total-vs-active for s-code-20m: 18.73M total, 15.63M active — ~3.1M params
of free capacity from sparsity.

Routing health is enforced by two auxiliary losses (scaled by
`aux_loss_coef = 0.01` / `z_loss_coef = 0.001` in training):
load-balance `E·Σ(frac_e · prob_e)` (floor 1.0 at perfect uniformity) and
router z-loss `mean(logsumexp²)` for logit stability. Each expert serves at
most `ceil(real_tokens · 1.25 / E)` tokens; overflow tokens skip the FFN via
the residual (never dropped, never NaN). **Capacity counts real tokens
only** — counting padded tokens let padding change real outputs (caught by
the padding non-leakage test; would have been a train/serve skew in
production). Padding tokens get exactly zero FFN delta.

### 3e. QK-normalization

Shared RMSNorm over Q and K (one scale vector of dim `dh` each, 128 params
per layer) applied after projection, before scoring. Small-model training
stability finding from recent practice; costs almost nothing, ablated in
round 1 (removal candidate if it shows no retrieval delta).

### 3f. Stochastic depth

During training only, block `i` is skipped whole with probability ramping
`0 → 0.1` by depth. Regularizes deep-narrow stacks and shortens the
effective training depth (compute saved, generalization gained). Eval is
always the full stack — determinism test covers this.

### 3g. Fused attention kernel

`attn_impl = "sdpa"` (default) routes the identical math through
`F.scaled_dot_product_attention`: Q pre-scaled by `sqrt(dh/d)` so the fused
`QK^T/sqrt(dh)` equals our `Qc·Kc/sqrt(d)`, with C2P + P2C + window +
padding passed as a single bias tensor. Same softmax, same output —
`test_sdpa_matches_eager` asserts parity to 1e-5 with shared weights under a
ragged mask. The win is GPU-fused kernels on Kaggle (memory-efficient /
flash backends), not different math. `"eager"` remains for readability and
CPU debugging.

## 4. Sofuu-DPB bucketing

Signed distance `r = i − j`, `|r|` clamped to `D`. With `half = B/2 = 16`
and `exact = 8`: magnitudes `0..7` map to exact buckets; magnitudes `8..D`
map to log-spaced buckets `8..15`; sign selects the half
(`r < 0 → +16`). Anything beyond `D` saturates into the last bucket —
far distances never decay toward zero magnitude, which is the deliberate
fix for the fixed-slope fp-underflow failure reported for ALiBi-style
biases. Table init `N(0, 0.02²)` so content dominates at step zero.

## 5. Normalization

RMSNorm everywhere (post-embedding, 2× per block, pre-pool):
`RMSNorm(x) = w ⊙ x / sqrt(mean(x²) + eps)`, `eps = 1e-6`. Chosen over
LayerNorm to drop the mean-centering reduction pass per token — on our
CPU/C-SIMD inference path reductions dominate latency, not FLOPs.

## 6. Pooling and output

Masked mean over real tokens only (padding contributes exactly zero;
proven by test), then L2 normalization → 256-d unit vector.
`matryoshka(dim)` truncates to a prefix and renormalizes; the joint
multi-prefix training loss arrives with the training loop, which only
relies on this truncation contract.

## 7. Exact parameter tables (mechanically enforced)

Per-layer (d=256): attn 262,144 + QK-norm 128 + DPB 8,192 + FFN (f=768)
589,824 + norms 512 = dense **860,800** / MoE **862,848** (+router 2,048);
with f=512: dense **664,192** / MoE **666,240** (+router 2,048).

| Model | Emb | Trunk (dense+MoE) | Head norms | Total | Active | Budget |
|---|---|---|---|---|---|---|
| s-code-20m (direct, L12, f768) | 8,388,608 | 10,341,888 (6+6) | 512 | 18,731,008 | 15,634,432 | 20M ✓ |
| s-docs-15m (fact-128, L12, f768) | 4,227,072 | 10,341,888 (6+6) | 512 | 14,569,472 | 11,472,896 | 15M ✓ |
| s-mem-10m (fact-128, L8, f512) | 4,227,072 | 5,321,728 (4+4) | 512 | 9,549,312 | 8,173,056 | 10M ✓ |

## 8. Prior art (ideas used, nothing copied)

All code is original. The following published results informed the design
and are cited here so the lineage is auditable:

- Scaled dot-product attention — Vaswani et al., *Attention Is All You
  Need* (2017). The substrate.
- Bidirectional encoders; NSP is dead weight — Devlin et al., BERT (2019);
  Liu et al., RoBERTa (2019).
- Disentangled C2C/C2P/P2C; keep absolute positions out of the trunk — He
  et al., DeBERTa (2020).
- Fixed distance penalties extrapolate; watch fp-underflow in far heads —
  Press et al., ALiBi (2021); Schröder et al. (2025/26).
- Multiplicative rotary positions, per-layer injection — Su et al., RoFormer
  (2021). (Not adopted in v0; candidate for ablation round 2.)
- Deep-narrow beats shallow-wide; MHA:FFN ratio ≈ 0.4–0.6; norms dominate
  real latency — Sun et al., MobileBERT (2020).
- Embedding factorization — Lan et al., ALBERT (2020).
- Gated linear units help at fixed compute — Shazeer, GLU Variants (2020).
- Modern encoder recipe: RoPE + local/global alternation + no biases +
  unpadding, all ablation-backed — Warner et al., ModernBERT (2024).
- 3-stage embedding recipe (MLM → unsupervised InfoNCE → supervised
  finetune with hard negatives); 30% masking — Nussbaum et al., Nomic
  Embed (2024).
- Matryoshka nested dimensions — Kusupati et al. (2024).
- InfoNCE objective — van den Oord et al. (2019).
- Dropout-as-augmentation baseline — Gao et al., SimCSE (2022).
- Sparse MoE for embeddings: capacity without active-cost scaling; alternate
  dense/sparse upcycling; top-k routing + load-balance aux loss — Shazeer et
  al., Switch Transformer (2022); Fedus et al. (2022); Komatsuzaki et al.,
  upcycling (2023); Nussbaum & Duderstadt, Nomic Embed v2 (2025) — first
  general-purpose MoE embedder, our direct precedent for MoE-in-retrieval.
- Router z-loss for logit stability — Zoph et al., ST-MoE (2022).
- QK-normalization for small-model stability — Dehghani et al. (2023);
  adopted widely in recent small LMs.
- Stochastic depth as depth-regularization — Huang et al. (2016).
- Fused exact attention kernels (same softmax, less memory traffic) — Dao et
  al., FlashAttention (2022).

## 9. Ablation plan (what makes the design ours)

The harness exists and runs: `sofuu_arch/synth.py` (motif-retrieval task,
train motifs disjoint from held motifs, motif-level relevance) +
`sofuu_arch/ablate.py` (InfoNCE trainer, false-negative-free batches,
6-variant × N-seed grid, JSON reports in `reports/`). Usage:
`python3 -m sofuu_arch.ablate --fast` (smoke) or bare (full grid).

### 9a. First results (2026-10-10, proxy L4/d64 CPU, 150 steps, 4 seeds)

| Variant | R@1 | Δ vs full | Reading |
|---|---|---|---|
| full | 0.579±0.049 | — | learns (chance ≈ 0.08) |
| no-moe (all dense) | 0.629±0.025 | +0.050 | MoE shows NO benefit here |
| no-qknorm | 0.563±0.062 | −0.017 | inconclusive |
| no-stochdepth | 0.571±0.045 | −0.008 | inconclusive |
| no-position | 0.567±0.054 | −0.013 | inconclusive (earlier −0.083 was seed luck) |
| plain-ffn | 0.558±0.073 | −0.021 | inconclusive |

Verdict: at proxy scale on synthetic motifs, nothing is proven and
nothing is disproven — all deltas sit inside seed noise. MoE specifically
has no evidence of helping yet (dense ≥ MoE twice). Every component stays
a hypothesis until the pilot-scale run on real data. The harness's job is
to force that verdict mechanically, not to flatter the design.

Two harness bugs were found and fixed while getting here (both would have
silently corrupted verdicts): in-batch false negatives (same-motif pairs
pushed apart as negatives — the loss fought itself, R@1 ≈ chance) and
exact-index eval metrics (same-motif distractors scored as misses).
Batches are now motif-unique and metrics are motif-level.

### 9b. Remaining shootouts (pilot scale, real data, Kaggle)

Round 1 (v1 confirmation): (a) DPB vs fixed-slope vs learned absolute;
(b) gated vs plain at matched params [proxy: inconclusive]; (c)
disentangled 3-term vs content-only [proxy: inconclusive]; (d) local/global
vs full-global at 256; (e) attention-pool vs mean; (f) factorized vs direct
embeddings [proxy skips: style fixed to direct]; (g) MoE-alternate vs
all-dense at matched total params [proxy: dense ≥ MoE, go/no-go open];
(h) QK-norm on/off [proxy: inconclusive]; (i) stochastic depth on/off
[proxy: inconclusive].

Winners compose block v2; losers are documented with numbers.

## 10. Honesty appendix

- ModernBERT's recipe was validated at 149M+ params on ~2T tokens; we
  operate at ~1/10th the size on ~1/10000th the data. Nothing transfers
  by authority — only by our measurements.
- No retrieval number exists yet for any config. The budgets above are
  engineering constraints, not quality claims.
- MoE at 10–20M scale is the least-proven component: all MoE-embedding
  evidence (Nomic v2) sits at 300M+ params with 16k-batch training. Small
  experts + small batches risk router collapse that the aux loss can't fix —
  ablation (g) is a genuine go/no-go, with all-dense as the fallback that
  costs nothing to keep (one config flag).
- Footnote on the plain-FFN ablation: `ffn_kind="plain"` swaps only the
  dense layers; MoE experts stay gated-SiLU. The comparison is
  "gated-alternate-MoE vs plain-dense", not pure gated-vs-plain.
- RoPE, unpadding, and quantization are deferred to round 2-equivalent
  work, after v1 proves the trunk.

## 11. Audit record (2026-10-10, model + training loop)

Method: re-read all eight modules, static checks (clean compile, no
retain_graph/bare-excepts), runtime leak probes (live-tensor counts via gc
over 60 train steps + 30 eval/train toggle cycles + 40 padding-variation
runs + checkpoint-tuple backward). Result: **no memory leaks** (0 tensor
growth second-window; padding outputs bit-stable). Bugs found and fixed,
each pinned by a regression test:

1. **[Critical] LR was squared.** The LambdaLR closure returned absolute
   lrs, so the effective peak was lr² (4e-8 instead of 2e-4) — every
   train.py run would silently not learn. Fixed (pure fractions) and
   pinned end-to-end (peak == base through a real optimizer). Ablation
   results unaffected (ablate.py uses a fixed AdamW, no scheduler).
2. **[Critical] `train.py` had no `__main__ guard.** The documented
   `torchrun -m sofuu_arch.train` exited 0 doing nothing — a wasted GPU
   session with zero signal. Fixed; subprocess entrypoint test asserts
   step logs appear.
3. **[Medium] `ablate.evaluate()` left the model in eval mode**, silently
   disabling stochastic depth after the first mid-training check. Fixed
   (mode save/restore) + pinned.
4. **[Low] Non-atomic checkpoints** (killed run = corrupt last.pt) and
   **opt_step reset on resume** (eval/save cadence restarted). Fixed
   (tmp+os.replace; opt_step stored/restored).
5. **[Trivial] Stale docstring, unused import, unused variable.**

Verified post-fix: real smoke run shows the warmup ramp (6.7e-06 →
2e-4) and falling loss (7.9 → 4.8 over 60 steps on random motifs).
Suite: 30/30 green.
