"""SofuuBlock: pre-norm disentangled attention + gated FFN (original code).

Block layout (pre-norm, no biases anywhere):

    h = x + Attn(RMSNorm(x))     # disentangled C2C + C2P + P2C
    y = h + FFN(RMSNorm(h))      # gated-SiLU FFN

Global layers attend to every token; local layers attend within a sliding
window of `local_window` on each side (padding positions are always masked).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpb import SofuuDPB
from .rmsnorm import RMSNorm


class DisentangledAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dpb_buckets: int,
        dpb_max_distance: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        attn_impl: str = "sdpa",
        position: str = "dpb",
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(dim)
        # SDPA scales by 1/sqrt(head_dim) internally; pre-scale Q so the
        # fused path computes exactly Qc.Kc/sqrt(dim).
        self.sdpa_q_scale = math.sqrt(self.head_dim / dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        assert position in ("dpb", "none")
        self.position = position
        self.dpb = SofuuDPB(dpb_buckets, dim, dpb_max_distance) if position == "dpb" else None
        self.dropout = nn.Dropout(dropout)
        assert attn_impl in ("eager", "sdpa")
        self.attn_impl = attn_impl

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        is_global: bool,
        local_window: int,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, n, self.heads, self.head_dim))
        v = self.v_proj(x).view(b, n, self.heads, self.head_dim)
        if self.position == "dpb":
            rel = self.dpb(n, x.device).view(n, n, self.heads, self.head_dim)
            # C2P + P2C position terms, computed per head.
            pos_bias = torch.einsum("bihd,ijhd->bhij", q, rel) + torch.einsum(
                "ijhd,bjhd->bhij", rel, k
            )
        else:
            # Content-only ablation: no position terms at all.
            pos_bias = torch.zeros(b, self.heads, n, n, device=x.device, dtype=q.dtype)
        mask = self._build_mask(b, n, x.device, key_padding_mask, is_global, local_window)

        if self.attn_impl == "sdpa":
            # Fused path: SDPA computes softmax(QK^T/sqrt(dh) + bias)V with
            # the position/window/padding terms passed as one bias tensor.
            # Mathematically identical to the eager path below.
            qt = (q * self.sdpa_q_scale).transpose(1, 2)  # (b, h, n, dh)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            bias = (pos_bias + mask).to(qt.dtype)
            out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias, dropout_p=self.dropout.p if self.training else 0.0)
            out = out.transpose(1, 2).reshape(b, n, self.dim)
        else:
            c2c = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale
            scores = c2c + pos_bias + mask
            weights = self.dropout(F.softmax(scores, dim=-1))
            out = torch.einsum("bhij,bjhd->bihd", weights, v)
            out = out.reshape(b, n, self.dim)
        return self.o_proj(out)

    def _build_mask(
        self,
        b: int,
        n: int,
        device: torch.device,
        key_padding_mask: torch.Tensor | None,
        is_global: bool,
        local_window: int,
    ) -> torch.Tensor:
        neg = torch.finfo(torch.float32).min
        mask = torch.zeros(b, 1, n, n, device=device)
        if not is_global:
            pos = torch.arange(n, device=device)
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
            window = (dist <= local_window).to(mask.dtype)
            mask = mask + (1.0 - window).unsqueeze(0).unsqueeze(0) * neg
        if key_padding_mask is not None:
            # key_padding_mask: (b, n) True == real token.
            allowed = key_padding_mask.to(mask.dtype).view(b, 1, 1, n)
            mask = mask + (1.0 - allowed) * neg
        return mask


class GatedFFN(nn.Module):
    """Gated-SiLU feed-forward: down(silu(gate(x)) * up(x)). No biases."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class PlainFFN(nn.Module):
    """Standard 2-layer SiLU FFN with 1.5x inner dim: 2*d*1.5f == 3*d*f.

    Parameter-matched to GatedFFN by construction, so gated-vs-plain
    ablations compare architecture, not size.
    """

    def __init__(self, dim: int, plain_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(dim, plain_dim, bias=False)
        self.down = nn.Linear(plain_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.up(x))))


class SofuuBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        dpb_buckets: int,
        dpb_max_distance: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        qk_norm: bool = True,
        attn_impl: str = "sdpa",
        position: str = "dpb",
        ffn_kind: str = "gated",
        plain_ff_dim: int | None = None,
        moe: nn.Module | None = None,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(dim, eps)
        self.attn = DisentangledAttention(
            dim, heads, dpb_buckets, dpb_max_distance, dropout,
            qk_norm=qk_norm, attn_impl=attn_impl, position=position,
        )
        self.norm_ffn = RMSNorm(dim, eps)
        # moe module is built by the encoder (avoids a block<->moe import cycle).
        self.is_moe = moe is not None
        if self.is_moe:
            self.ffn = moe
        elif ffn_kind == "gated":
            self.ffn = GatedFFN(dim, ffn_dim, dropout)
        else:
            self.ffn = PlainFFN(dim, plain_ff_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        is_global: bool,
        local_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (output, aux_balance, z_loss); dense FFN reports zeros."""
        x = x + self.attn(self.norm_attn(x), key_padding_mask, is_global, local_window)
        h = self.norm_ffn(x)
        if not self.is_moe:
            x = x + self.ffn(h)
            zero = torch.zeros((), device=x.device)
            return x, zero, zero
        delta, aux, z = self.ffn(h, key_padding_mask)
        return x + delta, aux, z
