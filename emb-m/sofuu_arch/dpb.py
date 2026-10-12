"""Sofuu-DPB: log-bucketed relative-position tables (original design).

Each layer owns a table E of `buckets` vectors of dim `d`. For a query at
position i and key at position j, the signed distance r = i - j is mapped
to a bucket b(r):

  - small |r| map to exact buckets (exact local order preserved);
  - large |r| map to logarithmically spaced buckets (distant order blurred).

The disentangled score adds two position terms to content-content attention:

    A[i, j] = Qc[i] . Kc[j] / sqrt(d)   (content -> content)
            + Qc[i] . E[b(i - j)]        (content -> position)
            + E[b(i - j)] . Kc[j]        (position -> content)

No absolute position embeddings exist anywhere in the trunk. The table is
tiny (buckets x d params per layer) and shared across heads. Log spacing
avoids the fixed-slope fp-underflow failure mode reported for ALiBi-style
biases: far distances saturate into the last bucket instead of decaying
toward zero magnitude.
"""

import math

import torch
import torch.nn as nn


def bucket_distance(distance: torch.Tensor, buckets: int, max_distance: int) -> torch.Tensor:
    """Map signed integer distances to bucket ids in [0, buckets).

    Bucket 0..half cover exact small distances; the rest cover log-spaced
    ranges up to max_distance; anything beyond saturates at the last bucket.
    Sign is folded in: negative and positive distances share magnitude
    buckets in separate halves (bidirectional encoder needs the sign).
    """
    half = buckets // 2
    sign = (distance < 0).to(torch.long) * half
    mag = distance.abs().clamp(max=max_distance)
    # Exact buckets for 0 .. exact_end-1, log buckets beyond.
    exact_end = half // 2
    small = mag < exact_end
    large_bucket = exact_end + (
        (torch.log(mag.to(torch.float) / exact_end + 1e-9)
         / math.log(max_distance / exact_end + 1e-9)
         * (half - exact_end)).to(torch.long)
    ).clamp(max=half - 1)
    mag_bucket = torch.where(small, mag.to(torch.long), large_bucket)
    return (sign + mag_bucket).clamp(max=buckets - 1)


class SofuuDPB(nn.Module):
    """Per-layer relative-position table + bucketed bias lookup."""

    def __init__(self, buckets: int, dim: int, max_distance: int = 256):
        super().__init__()
        self.buckets = buckets
        self.max_distance = max_distance
        # Small init so position terms start near zero and content dominates.
        self.table = nn.Parameter(torch.randn(buckets, dim) * 0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return relative vectors R[i, j] of shape (seq_len, seq_len, dim)."""
        pos = torch.arange(seq_len, device=device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # R[i, j] uses r = i - j.
        bucket = bucket_distance(dist, self.buckets, self.max_distance)
        return self.table[bucket]
