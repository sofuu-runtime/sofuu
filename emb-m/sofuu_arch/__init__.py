"""Sofuu embedding architecture package (clean-room, original code).

Modules:
    config   Model configurations and parameter-budget validation.
    params   Exact, dependency-free parameter counter (mirrors the torch modules).
    rmsnorm  RMS normalization layer.
    dpb      Sofuu-DPB log-bucketed relative-position bias tables.
    block    SofuuBlock: disentangled attention + gated FFN, local/global layers.
    encoder  Full encoder: embeddings, block stack, masked mean pooling, L2 out.
"""

from .config import ModelConfig, PRESETS, get_preset
from .encoder import SofuuEncoder
from .moe import SparseMoEFFN
from .params import count_active_params, count_params, budget_report

__all__ = [
    "ModelConfig",
    "PRESETS",
    "get_preset",
    "SofuuEncoder",
    "SparseMoEFFN",
    "count_params",
    "count_active_params",
    "budget_report",
]
