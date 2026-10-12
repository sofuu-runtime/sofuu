"""Contrastive losses for multi-GPU training (original code).

info_nce_gathered:
    Each rank encodes its local (queries, docs). Doc vectors (and their
    hard negatives) are all_gathered across ranks -- differentiable, so
    gradients flow back to every rank's encoder. Every query then contrasts
    against the GLOBAL candidate pool:

        candidates = [all positives (world*B)] + [all hard negatives]
        label[q]   = global index of q's own positive

    On 2xT4 this multiplies the negative pool by 2 at zero extra memory
    per rank beyond the gather buffer. With gradient accumulation the pool
    is per micro-step (standard practice; negatives do not cross micro-steps).

matryoshka_loss:
    Joint loss over output prefixes (default 256/128/64 with weights
    1.0/0.5/0.25): truncated+renormalized vectors get their own InfoNCE
    term, so every serving size is trained, not just sliced post-hoc.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    world = dist.get_world_size()
    if world == 1:
        return t
    out = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(out, t)  # differentiable: grads flow back to each rank.
    return torch.cat(out, dim=0)


def info_nce_gathered(
    q: torch.Tensor,           # (B, d) local query vecs (L2-normalized)
    d_pos: torch.Tensor,       # (B, d) local positive vecs
    d_hard: torch.Tensor | None,  # (B, H, d) or None
    tau: float,
) -> tuple[torch.Tensor, int]:
    """Returns (loss, global_negative_count)."""
    all_pos = _gather(d_pos)                       # (world*B, d)
    cands = [all_pos]
    if d_hard is not None and d_hard.numel() > 0:
        cands.append(_gather(d_hard.reshape(-1, d_hard.shape[-1])))
    cand = torch.cat(cands, dim=0)                 # (world*B + world*B*H, d)
    logits = (q @ cand.T) / tau
    rank = dist.get_rank() if dist.is_initialized() else 0
    base = d_pos.shape[0]
    labels = torch.arange(q.shape[0], device=q.device) + rank * base
    loss = F.cross_entropy(logits, labels)
    return loss, cand.shape[0] - q.shape[0]


def truncate_renorm(vec: torch.Tensor, dim: int) -> torch.Tensor:
    return F.normalize(vec[:, :dim], p=2, dim=-1)


def matryoshka_loss(
    q_full: torch.Tensor,
    d_pos_full: torch.Tensor,
    d_hard_full: torch.Tensor | None,
    dims: tuple,
    weights: tuple,
    tau: float,
) -> tuple[torch.Tensor, int]:
    total, negs = None, 0
    for dim, w in zip(dims, weights):
        loss, negs = info_nce_gathered(
            truncate_renorm(q_full, dim),
            truncate_renorm(d_pos_full, dim),
            truncate_renorm(d_hard_full, dim) if d_hard_full is not None else None,
            tau,
        )
        total = w * loss if total is None else total + w * loss
    return total, negs
