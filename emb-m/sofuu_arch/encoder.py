"""SofuuEncoder: embeddings, block stack, pooling, L2 output (original code).

Forward:

    ids (b, n) + padding mask -> token embeddings (direct or factorized)
      -> RMSNorm -> dropout -> L x SofuuBlock (alternating local/global)
      -> RMSNorm -> masked mean pool over real tokens -> L2 normalize

`matryoshka(dim)` truncates the normalized vector to a prefix; Matryoshka
training (joint loss over prefixes) arrives with the training loop -- this
module only guarantees the truncation contract.
"""

import torch
import torch.nn as nn

from .block import SofuuBlock
from .config import ModelConfig
from .moe import SparseMoEFFN
from .rmsnorm import RMSNorm


class SofuuEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        if cfg.embed_style == "direct":
            self.tok_emb = nn.Embedding(cfg.vocab_size, d)
            self.emb_proj = None
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_factor_dim)
            self.emb_proj = nn.Linear(cfg.embed_factor_dim, d, bias=False)
        self.emb_norm = RMSNorm(d, cfg.rms_eps)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                SofuuBlock(
                    dim=d,
                    heads=cfg.num_heads,
                    ffn_dim=cfg.ffn_dim,
                    dpb_buckets=cfg.dpb_buckets,
                    dpb_max_distance=cfg.dpb_max_distance,
                    dropout=cfg.dropout,
                    eps=cfg.rms_eps,
                    qk_norm=cfg.qk_norm,
                    attn_impl=cfg.attn_impl,
                    position=cfg.position,
                    ffn_kind=cfg.ffn_kind,
                    plain_ff_dim=cfg.plain_ff_dim,
                    moe=(
                        SparseMoEFFN(
                            dim=d,
                            expert_ff_dim=cfg.expert_ff_dim,
                            num_experts=cfg.num_experts,
                            capacity_factor=cfg.moe_capacity_factor,
                            dropout=cfg.dropout,
                        )
                        if cfg.is_moe_layer(i)
                        else None
                    ),
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm(d, cfg.rms_eps)

    def forward(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (unit_vectors, aux_balance_loss, router_z_loss).

        aux/z are zero for fully dense configs; the training loop scales
        them by cfg.aux_loss_coef / cfg.z_loss_coef. Stochastic depth
        (drop-path) applies per block during training only.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)

        x = self.tok_emb(ids)
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        x = self.emb_dropout(self.emb_norm(x))

        aux_total = torch.zeros((), device=x.device)
        z_total = torch.zeros((), device=x.device)
        for i, blk in enumerate(self.blocks):
            if self.training and torch.rand(()) < self.cfg.drop_path_rate(i):
                continue  # stochastic depth: skip the whole block.
            x, aux, z = blk(
                x,
                key_padding_mask=attention_mask,
                is_global=self.cfg.is_global_layer(i),
                local_window=self.cfg.local_window,
            )
            aux_total = aux_total + aux
            z_total = z_total + z
        x = self.final_norm(x)

        # Masked mean pool: padding contributes exactly zero.
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return nn.functional.normalize(pooled, p=2, dim=-1), aux_total, z_total

    def matryoshka(self, vec: torch.Tensor, dim: int) -> torch.Tensor:
        """Truncate to a prefix dim and renormalize (MRL inference contract)."""
        assert dim <= self.cfg.hidden_dim, "prefix exceeds output dim"
        return nn.functional.normalize(vec[:, :dim], p=2, dim=-1)
