# Sofuu Embeddings — `emb-m`

Clean-room, domain-specialized embedding models for the Sofuu runtime.
Every line of model code in this folder is written from scratch for this
project. Published mathematics and measured findings from the literature
inform the design (cited as prior art in `docs/ARCHITECTURE.md`); no
third-party code, weights, or prose are reused.

## Status

**Architecture v0 implemented, tested, unmeasured.** The three model
configs are specified, built in PyTorch, and covered by tests (8/8 green).
No training has run yet — every performance expectation is a hypothesis
until the pilot numbers exist.

## Layout

```
emb-m/
  README.md                This file.
  docs/
    ARCHITECTURE.md        Full block spec: equations, shapes, budgets,
                           prior-art citations, ablation plan.
    TRAINING.md            2xT4 operator guide: launch, batch math, memory,
                           data contracts, failure modes.
  sofuu_arch/              The model package (PyTorch, CPU-first).
    config.py              ModelConfig + presets (s-code-20m, s-docs-15m,
                           s-mem-10m) with construction-time validation.
    params.py              Exact pure-Python parameter counter (no torch needed).
    rmsnorm.py             RMSNorm layer.
    dpb.py                 Sofuu-DPB log-bucketed relative-position tables.
    block.py               SofuuBlock: disentangled attention + gated/plain FFN,
                           alternating local/global + dense/MoE layers.
    moe.py                 SparseMoEFFN: top-1 router, aux + z-loss, capacity.
    encoder.py             Embeddings, block stack, drop-path, masked mean pool,
                           L2 output, Matryoshka truncation contract.
    synth.py               Motif-retrieval task for ablations.
    ablate.py              Component ablation runner (6 variants x N seeds).
    data.py                JSONL pair loading, collate, motif smoke records.
    losses.py              Distributed InfoNCE (cross-GPU gathered negatives),
                           hard negatives, Matryoshka joint loss.
    train.py               DDP/NCCL trainer: fp16 AMP, grad accum, checkpoint /
                           resume, retrieval eval. See docs/TRAINING.md.
  tests/
    test_model.py          Budget agreement, shapes, mask correctness.
    test_ablation.py       Harness smoke test.
    test_train.py          Losses, checkpoint round-trip, CPU train smoke.
```

## Quickstart

```bash
cd emb-m
python3 -m unittest discover -s tests          # run all tests (needs torch)
python3 -c "
from sofuu_arch import get_preset, budget_report
print(budget_report(get_preset('s-code-20m')))"
```

## Model family (proposed specs, not results)

| Model | Task | Total params (exact) | Active params | Budget |
|---|---|---|---|---|
| s-code-20m | Code retrieval | 18,731,008 | 15,634,432 | 20M |
| s-docs-15m | Technical-doc retrieval | 14,569,472 | 11,472,896 | 15M |
| s-mem-10m | Agent-memory retrieval | 9,549,312 | 8,173,056 | 10M |

Totals are pinned in `tests/test_model.py` — any architecture change that
moves a parameter fails the test until the spec is deliberately revised.
"Active" is what one forward pass touches (top-1 MoE activates 1 of 8
experts); the gap is free capacity from sparsity.

## Design in one paragraph
A deep-narrow bidirectional encoder (d=256, 8–12 layers) with factorized or
direct token embeddings, disentangled content/position attention over our
own log-bucketed relative tables (Sofuu-DPB, no absolute positions in the
trunk), alternating local-128/global layers, **sparse MoE feed-forwards on
alternate layers (8 experts, top-1 routing, same total params as dense)**,
QK-normalization, stochastic depth, gated-SiLU FFNs, RMSNorm pre-norms,
zero bias terms anywhere, fused-SDPA attention kernel (exact same math as
the eager path, tested for parity), masked mean pooling, L2-normalized
256-d output with Matryoshka 256/128/64 prefixes. Full rationale, equations,
and the ablation shootouts that will confirm or kill each choice live in
`docs/ARCHITECTURE.md`.

## Data & legal doctrine

Training data allowlist: public domain, CC0, MIT, BSD, Apache-2.0, plus
CC-BY-4.0 only when needed and only with a complete per-record attribution
chain. CodeSearchNet is **rejected for training** (unverifiable mixed
per-repo licenses); the pilot corpus is self-built from license-filtered
repos with per-record provenance (repo URL, commit, license evidence).
Full doctrine in the project spec (§13–§16); the machine-readable license
gate arrives with the dataset pipeline (Phase 2).

## Ablation harness (does each part earn its place?)

```bash
python3 -m sofuu_arch.ablate --fast   # smoke: full vs no-moe, 1 seed
python3 -m sofuu_arch.ablate          # full grid: 6 variants x 2 seeds
```

`synth.py` builds a motif-retrieval task (train/held motifs disjoint,
motif-level relevance, false-negative-free batches); `ablate.py` trains
tiny proxies with identical data/seeds and reports R@1/R@5/MRR deltas vs
full v1 plus a JSON report in `reports/`. First verdict (2026-10-10):
the task learns (R@1 0.58 vs ~0.08 chance) but all component deltas sit
inside seed noise — **MoE shows no benefit yet (dense ≥ MoE)**. Nothing is
proven; the pilot-scale run on real data decides. Details in
`docs/ARCHITECTURE.md` §9.

## Roadmap

1. **Phase 1 (current):** architecture v1 ✅ · ablation harness ✅ ·
   distributed trainer ✅ (`train.py`: DDP/NCCL, fp16 AMP, gathered
   negatives, checkpoint/resume — 26/26 tests green) · license gate ·
   BPE trainer · ablation harness skeleton.
2. **Phase 2:** self-built pilot corpus (25–100k verified code pairs),
   repo-grouped splits, quality report with real counts.
3. **Phase 3:** ablation shootouts → block v1 → pilot training on Kaggle
   free-tier GPU → `.sem` export + Rust parity port.
4. **Phase 4:** held-out evaluation vs BM25/hash-768, failure analysis,
   go/no-go on expansion (Docs, Memory).
