# Training on 2× T4 — operator guide

Target: one box, two Tesla T4-16GB (e.g. Kaggle Notebooks GPU ×2), NCCL DDP,
single command. Everything below also degrades cleanly to 1 GPU (gather is a
no-op) and to CPU (no AMP) for smoke tests.

## 1. Launch

```bash
cd emb-m
pip install torch --index-url https://download.pytorch.org/whl/cu121  # T4 CUDA build

# smoke (no data files, ~1 min, proves the whole path incl. checkpointing)
torchrun --nproc_per_node=2 -m sofuu_arch.train --smoke \
  --preset s-mem-10m --steps 50 --micro-batch 64 --out runs/smoke

# real pilot (s-code-20m, 100k pairs)
torchrun --nproc_per_node=2 -m sofuu_arch.train \
  --preset s-code-20m \
  --train data/code_pairs.jsonl --eval data/code_eval.json \
  --micro-batch 512 --accum 2 --steps 20000 \
  --lr 2e-4 --warmup 500 --tau 0.02 \
  --out runs/code-v1
```

Resume after interruption (Kaggle 12h limit): re-run with
`--resume runs/code-v1/last.pt` — model, optimizer, scheduler, scaler, and
all RNG states restore; the sampler re-shuffles from the next epoch
(the only non-bitwise-exact piece; noted, accepted).

## 2. Batch math (why these defaults)

Contrastive quality scales with negatives-per-query. Per optimizer step:

```
negatives/query = micro_batch × world − 1  (+ H hard negs each)
              = 512 × 2 − 1 = 1023 in-batch (+ hard negatives)
optimizer batch = 512 × 2 × 2 (accum) = 2048 pairs/step
```

20k steps × 2048 = ~41M pair-views ≈ full pass structure for a 250k-pair
pilot repeated over ~160 epochs — tune `--steps` to your corpus (aim for
each pair to be seen with many different negative contexts).

## 3. Memory budget (T4-16GB, s-code-20m, seq 256, fp16)

| Item | Size |
|---|---|
| Weights (fp16) + master fp32 + AdamW states | ~0.3 GB |
| Activations @ micro 512 (scores dominate: B·h·n²·2B × 12 lyr) | ~4 GB |
| Gather buffers + hard negs (H ≤ 7) | < 1 GB |
| Headroom (NCCL, CUDA ctx, eval) | ~2 GB |

Fits comfortably. If you raise seq length or H, first add
`--grad-checkpoint` (activation checkpointing, ~30% slower, halves
activation memory; auto-disables stochastic depth), then lower
`--micro-batch` and raise `--accum` to hold the negative count.

## 4. What the script does per step

```
rank r: encode queries + positives + hard negs (fp16 AMP)
     → all_gather doc vecs across both GPUs (differentiable)
     → InfoNCE per Matryoshka prefix (256/128/64, weights 1/.5/.25)
     → + 0.01·aux_balance + 0.001·router_z
     → /accum → backward (no_sync on accum steps)
optimizer step: clip grad 1.0 → AdamW → linear warmup+decay → zero grad
rank 0: log JSONL (loss/lr/negatives/tok-s) · eval R@1/R@5/MRR · best.pt/last.pt
```

`find_unused_parameters=True` is set because idle MoE experts may see zero
tokens on a micro-step — without it DDP would error out. Cost: a small
all-reduce overhead per step; acceptable at our model size.

## 5. Throughput expectations (measured on real runs, fill in)

| Setup | tok/s | 20k steps ETA |
|---|---|---|
| 2× T4, s-code-20m, mb512/acc2 | _measure_ | _measure_ |
| 2× T4, s-mem-10m, mb512/acc2 | _measure_ | _measure_ |

Do not quote ETAs until a 100-step timing run fills this table.

## 6. Data contracts

Train JSONL (one object/line):
`{"query": [...ids], "positive": [...ids], "hard_negatives": [[...ids]]?}`

Eval JSON:
`{"queries": [[...ids]], "pool": [[...ids]], "positives": [[pool_idx, ...]]}`
(multi-positive supported; metrics are motif-style any-hit R@k + MRR).

Ids are pre-tokenized ints — the BPE trainer plugs in upstream of this
file format and nothing here changes.

## 7. Failure modes to watch

- **Router collapse** (`final_aux` stuck high in logs): raise `--tau`? No —
  raise data diversity first; aux loss only balances, it can't create signal.
- **Loss NaN under fp16**: GradScaler usually rides through; persistent NaN
  → lower `--lr`, check for corrupt records (empty sequences after
  truncation are masked, not crashed — but verify).
- **OOM on eval**: `encode_all` chunks at 512; lower via code constant if
  pool is huge (stream from disk instead — not yet implemented).
- **NCCL timeout on Kaggle**: set `NCCL_P2P_DISABLE=1` if the two T4s lack
  NVLink/P2P; slight comm slowdown, training unaffected.
