"""Exact parameter counter (pure Python, no torch).

Mirrors the torch modules one-to-one so budgets can be checked without
instantiating a model. Any architecture change MUST update both this file
and the corresponding module; `tests/test_model.py` asserts they agree.
"""

from .config import ModelConfig


def count_layer(cfg: ModelConfig, moe: bool = False) -> dict:
    d = cfg.hidden_dim
    attn_qkvo = 4 * d * d  # Q, K, V, O projections, no biases.
    qk_norm = 2 * (d // cfg.num_heads) if cfg.qk_norm else 0  # shared Q/K RMSNorm.
    dpb_table = cfg.dpb_buckets * d if cfg.position == "dpb" else 0
    if moe:
        # E experts of (ffn_dim // E) inner dim == dense ffn_dim in total,
        # plus the router. Active compute is ~1/E of the dense FFN.
        e_ff = cfg.expert_ff_dim
        ffn = cfg.num_experts * (3 * d * e_ff)
        router = d * cfg.num_experts
    else:
        ffn = 3 * d * cfg.ffn_dim  # gate, up, down projections, no biases.
        router = 0
    norms = 2 * d  # RMSNorm before attn + before FFN (must match SofuuBlock).
    total = attn_qkvo + qk_norm + dpb_table + ffn + router + norms
    return {
        "attn_qkvo": attn_qkvo,
        "qk_norm": qk_norm,
        "dpb_table": dpb_table,
        "ffn": ffn,
        "router": router,
        "norms": norms,
        "total": total,
    }


def count_params(cfg: ModelConfig) -> dict:
    v, d = cfg.vocab_size, cfg.hidden_dim
    if cfg.embed_style == "direct":
        embeddings = v * d
    else:
        e = cfg.embed_factor_dim
        embeddings = v * e + e * d  # factor table + projection, no bias.

    per_layer, per_moe = count_layer(cfg, False), count_layer(cfg, True)
    n_moe = sum(1 for i in range(cfg.num_layers) if cfg.is_moe_layer(i))
    n_dense = cfg.num_layers - n_moe
    trunk = per_layer["total"] * n_dense + per_moe["total"] * n_moe
    # Post-embedding RMSNorm + final RMSNorm before pooling.
    head_norms = 2 * d

    total = embeddings + trunk + head_norms
    return {
        "embeddings": embeddings,
        "per_layer": per_layer,
        "per_moe_layer": per_moe,
        "n_dense": n_dense,
        "n_moe": n_moe,
        "trunk": trunk,
        "head_norms": head_norms,
        "total": total,
        "budget": cfg.param_budget,
        "within_budget": total <= cfg.param_budget,
    }


def count_active_params(cfg: ModelConfig) -> int:
    """Params touched by one forward pass (top-1 MoE activates 1 expert)."""
    c = count_params(cfg)
    dense, moe = c["per_layer"], c["per_moe_layer"]
    # MoE layer active cost: dense parts + ONE expert + router.
    e_ff = cfg.expert_ff_dim
    active_moe = (
        moe["attn_qkvo"] + moe["qk_norm"] + moe["dpb_table"]
        + 3 * cfg.hidden_dim * e_ff + moe["router"] + moe["norms"]
    )
    return c["embeddings"] + dense["total"] * c["n_dense"] + active_moe * c["n_moe"] + c["head_norms"]


def budget_report(cfg: ModelConfig) -> str:
    c = count_params(cfg)
    lines = [
        f"model: {cfg.name}  budget: {c['budget']:,}  total: {c['total']:,}"
        f"  within_budget={c['within_budget']}",
        f"  embeddings:  {c['embeddings']:,}",
        f"  trunk:       {c['trunk']:,}  ({c['n_dense']} dense x {c['per_layer']['total']:,}"
        f" + {c['n_moe']} MoE x {c['per_moe_layer']['total']:,})",
        f"    attn_qkvo: {c['per_layer']['attn_qkvo']:,}",
        f"    qk_norm:   {c['per_layer']['qk_norm']:,}",
        f"    dpb_table: {c['per_layer']['dpb_table']:,}",
        f"    ffn:       {c['per_layer']['ffn']:,}",
        f"    router:    {c['per_moe_layer']['router']:,}",
        f"    norms:     {c['per_layer']['norms']:,}",
        f"  head_norms:  {c['head_norms']:,}",
    ]
    return "\n".join(lines)
