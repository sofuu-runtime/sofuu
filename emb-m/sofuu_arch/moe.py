"""Sparse Mixture-of-Experts FFN with top-1 routing (original code).

Each MoE layer holds E small gated-SiLU experts (inner dim = dense_ff / E,
so E experts cost exactly what one dense FFN costs in TOTAL params) plus a
linear router. Per token:

    logits = x @ W_router          (d -> E)
    e*     = argmax(logits)        (top-1)
    y      = gate_prob[e*] * expert[e*](x)

Two auxiliary losses keep routing healthy (added to the training loss by
the caller; this module only computes them):

- load-balance loss (Switch-style): E * sum_e(frac_tokens_e * mean_prob_e),
  minimized when tokens spread evenly. frac_tokens is computed from the
  discrete assignments (straight-through: use detached counts).
- router z-loss: mean(logsumexp(logits)^2), keeps logits small and the
  router confident-but-bounded (stability finding from large MoE practice).

Capacity: each expert accepts at most ceil(T * factor / E) tokens where T
is the batch token count. Overflow tokens skip the FFN (pure residual) --
no dropping, no NaNs, graceful degradation under imbalance.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import GatedFFN


class SparseMoEFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        expert_ff_dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_experts >= 2
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [GatedFFN(dim, expert_ff_dim, dropout) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (output, aux_balance_loss, z_loss).

        padding_mask: (b, n) bool, True == real token. Padding tokens are
        excluded from routing statistics AND from expert compute (their
        output delta is zero, so pooling is unaffected).
        """
        b, n, d = x.shape
        flat = x.reshape(-1, d)  # (T, d)
        if padding_mask is not None:
            real = padding_mask.reshape(-1).to(flat.dtype)  # (T,)
        else:
            real = torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype)

        logits = self.router(flat)  # (T, E)
        probs = F.softmax(logits, dim=-1)
        assign = probs.argmax(dim=-1)  # (T,) top-1 expert per token.

        # Capacity counts REAL tokens only: padding must not change how many
        # genuine tokens an expert serves (else padding leaks into outputs).
        t_real = max(1, int(real.sum().item()))
        capacity = max(1, math.ceil(t_real * self.capacity_factor / self.num_experts))

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            idx = (assign == e).nonzero(as_tuple=True)[0]
            # Keep real tokens up to capacity; padding never consumes slots.
            idx = idx[real[idx] > 0][:capacity]
            if idx.numel() == 0:
                continue
            gate = probs[idx, e].unsqueeze(-1)
            out[idx] = gate * expert(flat[idx])

        out = self.dropout(out).view(b, n, d)

        # --- auxiliary losses (padding excluded) ---
        with torch.no_grad():
            one_hot = F.one_hot(assign, self.num_experts).to(probs.dtype)
            frac = (one_hot * real.unsqueeze(-1)).sum(0) / real.sum().clamp(min=1.0)
        mean_prob = (probs * real.unsqueeze(-1)).sum(0) / real.sum().clamp(min=1.0)
        aux = (frac * mean_prob).sum() * self.num_experts
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return out, aux, z
