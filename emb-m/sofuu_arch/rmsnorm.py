"""RMS normalization layer (original implementation).

RMSNorm(x) = w * x / sqrt(mean(x^2) + eps). Drops LayerNorm's mean-centering,
which removes a reduction pass per token -- meaningful for our CPU/C-SIMD
inference port where reductions dominate latency, not FLOPs.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
