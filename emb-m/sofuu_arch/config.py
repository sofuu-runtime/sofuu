"""Model configurations and parameter-budget validation (pure Python, no torch).

Every Sofuu embedding model must remain within its declared parameter budget.
Configs are validated at construction: hidden dims must be multiples of 64
(CPU-SIMD co-design rule) and head dim must divide the hidden dim evenly.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int = 32768
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 12
    ffn_dim: int = 768
    # Token embedding style: "direct" (V x d) or "factorized" (V x e -> d).
    embed_style: str = "factorized"
    embed_factor_dim: int = 128
    # Sofuu-DPB relative-position table: buckets shared across layers.
    dpb_buckets: int = 32
    dpb_max_distance: int = 256
    # Local/global attention rhythm: every `global_every`-th layer is global
    # (0-indexed layers where (i + 1) % global_every == 0), rest are sliding
    # window with `local_window` radius on each side.
    global_every: int = 3
    local_window: int = 128
    max_seq_len: int = 256
    output_dim: int = 256
    # Matryoshka prefixes trained jointly (must be <= output_dim).
    matryoshka_dims: field(default_factory=lambda: (256, 128, 64)) = (256, 128, 64)
    dropout: float = 0.1
    rms_eps: float = 1e-6
    param_budget: int = 20_000_000
    # --- v1: sparsity / stability / speed ---
    # MoE rhythm: layers where is_moe_layer(i) is True use a sparse MoE FFN
    # (num_experts experts, top-k routing); all other layers stay dense.
    # "alternate" = every 2nd layer starting from index 1; "none" = dense.
    moe_pattern: str = "alternate"
    num_experts: int = 8
    moe_top_k: int = 1
    # Expert inner dim derives from ffn_dim so MoE layers cost the same
    # TOTAL params as dense ones: expert_ff = ffn_dim // num_experts.
    moe_capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01  # load-balance auxiliary loss weight.
    z_loss_coef: float = 0.001  # router z-loss weight (logit stability).
    qk_norm: bool = True  # RMSNorm on Q and K per head before scoring.
    stochastic_depth_max: float = 0.1  # drop-path ramp 0 -> max by depth.
    # Attention kernel: "eager" (exact, readable) or "sdpa" (same math via
    # fused SDPA with the position/window/padding terms passed as bias).
    attn_impl: str = "sdpa"
    # --- ablation kill-switches (every component must earn its place) ---
    # position: "dpb" (disentangled C2C+C2P+P2C) or "none" (content-only).
    position: str = "dpb"
    # ffn_kind: "gated" (SiLU-gated, 3 matmuls) or "plain" (SiLU 2-layer
    # with 1.5x inner dim -- matched params by construction: 2*d*1.5f=3df).
    ffn_kind: str = "gated"

    def __post_init__(self):
        if self.hidden_dim % 64 != 0:
            raise ValueError(f"hidden_dim must be a multiple of 64, got {self.hidden_dim}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.embed_style not in ("direct", "factorized"):
            raise ValueError(f"unknown embed_style: {self.embed_style}")
        if self.embed_style == "factorized" and self.embed_factor_dim >= self.hidden_dim:
            raise ValueError("factor dim must be smaller than hidden dim")
        if any(d > self.output_dim or d <= 0 for d in self.matryoshka_dims):
            raise ValueError("matryoshka dims must be within (0, output_dim]")
        if self.output_dim != self.hidden_dim:
            raise ValueError("v0 keeps output_dim == hidden_dim (pooling only, no projection)")
        if self.moe_pattern not in ("alternate", "none"):
            raise ValueError(f"unknown moe_pattern: {self.moe_pattern}")
        if self.ffn_dim % self.num_experts != 0:
            raise ValueError("ffn_dim must divide evenly by num_experts (MoE budget rule)")
        if self.moe_top_k != 1:
            raise ValueError("v1 supports top-1 routing only (validated path)")
        if self.attn_impl not in ("eager", "sdpa"):
            raise ValueError(f"unknown attn_impl: {self.attn_impl}")
        if self.position not in ("dpb", "none"):
            raise ValueError(f"unknown position: {self.position}")
        if self.ffn_kind not in ("gated", "plain"):
            raise ValueError(f"unknown ffn_kind: {self.ffn_kind}")
        if self.ffn_kind == "plain" and (self.ffn_dim * 3) % 2 != 0:
            raise ValueError("plain FFN needs ffn_dim*1.5 integral for param matching")

    @property
    def expert_ff_dim(self) -> int:
        return self.ffn_dim // self.num_experts

    @property
    def plain_ff_dim(self) -> int:
        return self.ffn_dim * 3 // 2

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    def is_global_layer(self, index: int) -> bool:
        return (index + 1) % self.global_every == 0

    def is_moe_layer(self, index: int) -> bool:
        return self.moe_pattern == "alternate" and index % 2 == 1

    def drop_path_rate(self, index: int) -> float:
        if self.num_layers <= 1:
            return 0.0
        return self.stochastic_depth_max * index / (self.num_layers - 1)


PRESETS = {
    # S-Code: code retrieval, 20M budget.
    "s-code-20m": ModelConfig(
        name="s-code-20m",
        embed_style="direct",
        num_layers=12,
        ffn_dim=768,
        param_budget=20_000_000,
    ),
    # S-Docs: technical-doc retrieval, 15M budget.
    "s-docs-15m": ModelConfig(
        name="s-docs-15m",
        embed_style="factorized",
        embed_factor_dim=128,
        num_layers=12,
        ffn_dim=768,
        param_budget=15_000_000,
    ),
    # S-Mem: agent-memory retrieval, 10M budget.
    "s-mem-10m": ModelConfig(
        name="s-mem-10m",
        embed_style="factorized",
        embed_factor_dim=128,
        num_layers=8,
        ffn_dim=512,
        param_budget=10_000_000,
    ),
}


def get_preset(name: str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
