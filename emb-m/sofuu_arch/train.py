"""Contrastive training, optimized for 2x T4 (single node) (original code).

Launch (Kaggle / any 2-GPU box):
    torchrun --nproc_per_node=2 -m sofuu_arch.train \\
        --preset s-code-20m --train pairs.jsonl --eval eval.json \\
        --micro-batch 512 --accum 2 --steps 20000 --out runs/code-v1

Why this shape for 2xT4-16GB:
- NCCL DDP, fp16 AMP + GradScaler (Turing Tensor Cores; T4 has no bf16).
- In-batch negatives are all_gathered across BOTH GPUs, so every query
  contrasts against 2x the local batch at no extra activation cost.
- Gradient accumulation (no_sync between micro-steps) pushes the effective
  batch further without touching the 16GB ceiling.
- Optional activation checkpointing trades ~30% speed for ~50% activation
  memory (forces stochastic-depth off: re-executed blocks must be
  deterministic under checkpointing).
- Pinned-memory loaders, cudnn benchmark, TF32 matmuls allowed.

Data formats:
- --train: JSONL {query:[ids], positive:[ids], hard_negatives:[[ids]]?}
- --eval:  JSON {queries:[[ids]...], pool:[[ids]...],
                  positives:[[pool_idx,...], ...]} (multi-positive ok)
- --smoke: skip files; train on synthetic motif pairs, eval on held motifs.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import data as data_mod
from .config import get_preset
from .encoder import SofuuEncoder
from .losses import matryoshka_loss
from .synth import SynthSpec


def linear_warmup_decay(t: int, warm: int, total: int) -> float:
    """LR multiplier fraction (LambdaLR scales the base lr by this).

    Returns 0..1 ramp over warm steps, then linear decay to 0 at total.
    MUST stay a pure fraction -- multiplying by the base lr here would
    square the effective rate (caught once by audit, pinned by test).
    """
    if t < warm:
        return (t + 1) / max(1, warm)
    p = (t - warm) / max(1, total - warm)
    return max(0.0, 1.0 - p)


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world, local


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def encode_all(model, ids_list: list, mask_list: list | None,
               device, chunk: int = 512) -> torch.Tensor:
    model.eval()
    vecs = []
    for i in range(0, len(ids_list), chunk):
        ids = torch.stack(ids_list[i : i + chunk]).to(device)
        mask = (torch.stack(mask_list[i : i + chunk]).to(device)
                if mask_list is not None else None)
        v, _, _ = unwrap(model)(ids, mask)
        vecs.append(v.float().cpu())
    model.train()
    return torch.cat(vecs, dim=0)


@torch.no_grad()
def evaluate_retrieval(model, eval_obj: dict, cfg, device, args) -> dict:
    """Generic multi-positive retrieval eval (rank 0 only)."""
    def pack(seqs):
        ids_list, mask_list = [], []
        for s in seqs:
            s = s[: args.max_seq_len]
            ids = torch.zeros(args.max_seq_len, dtype=torch.long)
            mask = torch.zeros(args.max_seq_len, dtype=torch.bool)
            ids[: len(s)] = torch.tensor(s, dtype=torch.long)
            mask[: len(s)] = True
            ids_list.append(ids)
            mask_list.append(mask)
        return ids_list, mask_list

    q_ids, q_mask = pack(eval_obj["queries"])
    p_ids, p_mask = pack(eval_obj["pool"])
    qv = encode_all(model, q_ids, q_mask, device)
    pv = encode_all(model, p_ids, p_mask, device)
    scores = qv @ pv.T
    r1 = r5 = mrr = 0.0
    n = len(q_ids)
    for i, pos in enumerate(eval_obj["positives"]):
        rel = torch.zeros(len(p_ids), dtype=torch.bool)
        rel[pos] = True
        order = scores[i].argsort(descending=True)
        ranked = rel[order]
        first = int(ranked.int().argmax())
        if ranked[0]:
            r1 += 1
        if ranked[:5].any():
            r5 += 1
        mrr += 1.0 / (first + 1)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": mrr / n}


def save_checkpoint(path: Path, step: int, opt_step: int, model, opt, sched,
                     scaler, best: float, rng_state, cuda_state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "opt_step": opt_step,
        "model": unwrap(model).state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best,
        "rng": rng_state,
        "cuda_rng": cuda_state,
    }, tmp)
    os.replace(tmp, path)  # atomic: a killed run never leaves a half file.


def main() -> None:
    ap = argparse.ArgumentParser(description="Sofuu contrastive trainer (2xT4 DDP)")
    ap.add_argument("--preset", default="s-code-20m")
    ap.add_argument("--train", default=None, help="JSONL pairs file")
    ap.add_argument("--eval", default=None, help="eval JSON file")
    ap.add_argument("--smoke", action="store_true", help="motif data, no files")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--micro-batch", type=int, default=512)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--mrl-weights", default="1.0,0.5,0.25")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    cfg = get_preset(args.preset)
    if args.grad_checkpoint and cfg.stochastic_depth_max > 0:
        if rank == 0:
            print("(!) grad-checkpoint needs deterministic blocks: "
                  "stochastic depth disabled for this run")
        cfg = dataclasses.replace(cfg, stochastic_depth_max=0.0)
    if rank == 0:
        eff_neg = args.micro_batch * world - 1
        print(f"preset={cfg.name} world={world} micro={args.micro_batch} "
              f"accum={args.accum} -> ~{eff_neg} in-batch negatives/query "
              f"(+hard negs), tau={args.tau}")

    # --- data ---
    if args.smoke:
        spec = SynthSpec(vocab_size=cfg.vocab_size, seq_len=args.max_seq_len)
        records = data_mod.motif_records(2048, args.seed, spec)
        eval_obj = None
    else:
        if not args.train:
            raise SystemExit("need --train pairs.jsonl or --smoke")
        records = data_mod.load_jsonl(args.train)
        eval_obj = json.load(open(args.eval)) if args.eval else None
    ds = data_mod.PairDataset(records, args.max_seq_len)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(ds, batch_size=args.micro_batch, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None,
                        collate_fn=data_mod.collate_pairs, drop_last=True)

    # --- model ---
    model = SofuuEncoder(cfg).to(device)
    if args.grad_checkpoint:
        from torch.utils.checkpoint import checkpoint as ckpt
        orig_forwards = {}
        for i, blk in enumerate(model.blocks):
            def make_fn(block):
                def fn(x, mask, is_global, local_window):
                    return block(x, mask, is_global, local_window)
                return fn
            orig_forwards[i] = blk.forward
            blk.forward = (lambda block: (lambda *a: ckpt(
                make_fn(block), *a, use_reentrant=False)))(blk)
    if world > 1:
        model = DDP(model, device_ids=[local] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)  # MoE experts may sit out a batch.
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0.01)
    total_opt_steps = math.ceil(args.steps / args.accum)
    warm = min(args.warmup, total_opt_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: linear_warmup_decay(t, warm, total_opt_steps))

    mrl_w = tuple(float(x) for x in args.mrl_weights.split(","))
    assert len(mrl_w) == len(cfg.matryoshka_dims), "mrl-weights must match preset dims"

    start_step, best, opt_step = 0, 0.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["rng"].cpu())
        if torch.cuda.is_available() and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start_step, best = ck["step"], ck["best"]
        opt_step = ck.get("opt_step", 0)  # pre-opt_step checkpoints restart cadence.
        if rank == 0:
            print(f"resumed from {args.resume} at step {start_step} (best R@1={best:.3f})")

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log.jsonl", "a") if rank == 0 else None

    # --- train ---
    model.train()
    t0 = time.time()  # opt_step comes from init (0) or resume above.
    it = iter(loader)
    for step in range(start_step, args.steps):
        try:
            q_ids, q_mask, d_ids, d_mask = next(it)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(sampler.epoch + 1)
            it = iter(loader)
            q_ids, q_mask, d_ids, d_mask = next(it)
        q_ids, q_mask = q_ids.to(device, non_blocking=True), q_mask.to(device, non_blocking=True)
        d_ids, d_mask = d_ids.to(device, non_blocking=True), d_mask.to(device, non_blocking=True)

        is_accum = (step + 1) % args.accum != 0
        ctx = model.no_sync() if (world > 1 and is_accum) else torch.enable_grad()
        with ctx, torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            qv, aux_q, z_q = model(q_ids, q_mask)
            B, H, L = d_ids.shape
            dv_all, aux_d, z_d = model(d_ids.reshape(B * H, L), d_mask.reshape(B * H, L))
            dv_all = dv_all.reshape(B, H, -1)
            main, n_negs = matryoshka_loss(qv, dv_all[:, 0], dv_all[:, 1:] if H > 1 else None,
                                           cfg.matryoshka_dims, mrl_w, args.tau)
            loss = (main + cfg.aux_loss_coef * (aux_q + aux_d)
                    + cfg.z_loss_coef * (z_q + z_d)) / args.accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if not is_accum:
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1

            if rank == 0 and (opt_step % 20 == 0 or opt_step == 1):
                dt = time.time() - t0
                toks = args.micro_batch * args.accum * world * args.max_seq_len * 2 / max(dt, 1e-9)
                msg = {"step": step + 1, "opt_step": opt_step, "loss": round(loss.item() * args.accum, 4),
                       "lr": round(sched.get_last_lr()[0], 7), "negatives": n_negs,
                       "tok_per_s": int(toks)}
                print(f"step {msg['step']} loss {msg['loss']} lr {msg['lr']} "
                      f"neg {msg['negatives']} {msg['tok_per_s']} tok/s", flush=True)
                logf.write(json.dumps(msg) + "\n")
                logf.flush()
                t0 = time.time()

            if rank == 0 and opt_step % max(1, args.eval_every) == 0 and eval_obj is not None:
                m = evaluate_retrieval(model, eval_obj, cfg, device, args)
                print(f"  eval R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
                      f"MRR={m['mrr']:.3f}", flush=True)
                if m["recall@1"] > best:
                    best = m["recall@1"]
                    save_checkpoint(out / "best.pt", step + 1, opt_step, model, opt, sched,
                                    scaler, best, torch.get_rng_state(),
                                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            if rank == 0 and opt_step % max(1, args.save_every) == 0:
                save_checkpoint(out / "last.pt", step + 1, opt_step, model, opt, sched,
                                scaler, best, torch.get_rng_state(),
                                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    if world > 1:
        dist.barrier()
    if rank == 0 and logf is not None:
        logf.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
