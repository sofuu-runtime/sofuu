"""Component ablation runner: does each part earn its place? (original code).

Trains small proxy models on the synthetic motif task (synth.py) — one run
per (variant, seed) — and reports retrieval deltas vs the full v1 config.
Every variant trains on IDENTICAL data with the same seeds; only the
architecture flags differ.

Variants:
    full          v1 as specified (MoE + QK-norm + stoch-depth + DPB + gated)
    no-moe        moe_pattern="none" (all dense)
    no-qknorm     qk_norm=False
    no-stochdepth stochastic_depth_max=0
    no-position   position="none" (content-only attention)
    plain-ffn     ffn_kind="plain" (param-matched 2-layer SiLU)

Usage:
    python3 -m sofuu_arch.ablate --fast            # smoke: 2 variants, 1 seed
    python3 -m sofuu_arch.ablate                   # full grid: 6 variants x 2 seeds
    python3 -m sofuu_arch.ablate --seeds 3 --steps 300 --variants full,no-moe

Report: JSON in emb-m/reports/ + printed mean +/- std table.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .encoder import SofuuEncoder
from .synth import SynthSpec, eval_set, train_pairs

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

VARIANTS = {
    "full": {},
    "no-moe": {"moe_pattern": "none"},
    "no-qknorm": {"qk_norm": False},
    "no-stochdepth": {"stochastic_depth_max": 0.0},
    "no-position": {"position": "none"},
    "plain-ffn": {"ffn_kind": "plain"},
}


def proxy_config(overrides: dict, spec: SynthSpec) -> ModelConfig:
    base = dict(
        name="proxy",
        vocab_size=spec.vocab_size,
        embed_style="direct",  # embedding style is not under ablation.
        hidden_dim=64,
        num_heads=2,
        num_layers=4,
        ffn_dim=128,
        max_seq_len=spec.seq_len,
        dpb_max_distance=spec.seq_len,
        local_window=spec.seq_len,
        output_dim=64,
        matryoshka_dims=(64, 32),
        dropout=0.0,
        attn_impl="eager",  # CPU-clean; kernel parity is tested separately.
    )
    base.update(overrides)
    return ModelConfig(**base)


def recall_at(scores: torch.Tensor, q_motifs: torch.Tensor,
              pool_motifs: torch.Tensor, k: int) -> float:
    """Motif-level recall: hit if ANY same-motif doc is in the top-k."""
    topk = scores.topk(min(k, scores.shape[1]), dim=1).indices
    relevant = (pool_motifs.unsqueeze(0) == q_motifs.unsqueeze(1))
    hits = relevant.gather(1, topk).any(dim=1).to(torch.float)
    return hits.mean().item()


def mrr(scores: torch.Tensor, q_motifs: torch.Tensor,
        pool_motifs: torch.Tensor) -> float:
    relevant = (pool_motifs.unsqueeze(0) == q_motifs.unsqueeze(1))
    order = scores.argsort(dim=1, descending=True)
    ranked_rel = relevant.gather(1, order)
    first = ranked_rel.int().argmax(dim=1)
    return (1.0 / (first.to(torch.float) + 1.0)).mean().item()


@torch.no_grad()
def evaluate(model: SofuuEncoder, queries, pool, q_motifs, pool_motifs) -> dict:
    was_training = model.training  # eval must not flip the training loop's mode.
    model.eval()
    qv, _, _ = model(queries)
    dv, _, _ = model(pool)
    scores = qv @ dv.T
    out = {
        "recall@1": recall_at(scores, q_motifs, pool_motifs, 1),
        "recall@5": recall_at(scores, q_motifs, pool_motifs, 5),
        "mrr": mrr(scores, q_motifs, pool_motifs),
    }
    model.train(was_training)
    return out


def train_variant(
    variant: str,
    seed: int,
    spec: SynthSpec,
    steps: int,
    batch_size: int = 8,
    lr: float = 3e-4,
    tau: float = 0.05,
    eval_every: int = 25,
) -> dict:
    torch.manual_seed(seed)
    cfg = proxy_config(VARIANTS[variant], spec)
    model = SofuuEncoder(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    tq, td, tm = train_pairs(spec, 512, seed + 1000)
    eq, pool, eq_motifs, pool_motifs = eval_set(spec, 60, 200, seed + 2000)
    # Motif -> pair indices, for false-negative-free batches: every batch
    # holds at most ONE pair per motif, so in-batch negatives are always
    # truly irrelevant.
    by_motif = {m: (tm == m).nonzero(as_tuple=True)[0] for m in tm.unique().tolist()}

    best = {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}
    final_aux = 0.0
    model.train()
    for step in range(steps):
        motifs = torch.randperm(len(by_motif))[:batch_size]
        idx = torch.stack([
            grp[torch.randint(0, len(grp), (1,))[0]]
            for m in motifs for grp in [by_motif[int(m)]]
        ])
        qv, aux, z = model(tq[idx])
        dv, aux_d, z_d = model(td[idx])
        logits = (qv @ dv.T) / tau
        target = torch.arange(len(idx))
        loss = F.cross_entropy(logits, target)
        loss = loss + cfg.aux_loss_coef * (aux + aux_d) + cfg.z_loss_coef * (z + z_d)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_aux = (aux + aux_d).item() / 2
        if (step + 1) % eval_every == 0 or step == steps - 1:
            m = evaluate(model, eq, pool, eq_motifs, pool_motifs)
            if m["recall@1"] > best["recall@1"]:
                best = m
    return {"variant": variant, "seed": seed, "steps": steps, **best, "final_aux": final_aux}


def summarize(results: list[dict]) -> str:
    by_variant: dict[str, list[dict]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)
    ref = sum(r["recall@1"] for r in by_variant.get("full", [])) / max(1, len(by_variant.get("full", [])))
    lines = [f"{'variant':<14}{'R@1 (mean±std)':<18}{'R@5':<8}{'MRR':<8}{'ΔR@1 vs full'}"]
    for name, rs in by_variant.items():
        r1 = [r["recall@1"] for r in rs]
        mean = sum(r1) / len(r1)
        std = (sum((v - mean) ** 2 for v in r1) / len(r1)) ** 0.5 if len(r1) > 1 else 0.0
        m = rs[0]
        # recompute means for r5/mrr across seeds
        r5 = sum(r["recall@5"] for r in rs) / len(rs)
        mr = sum(r["mrr"] for r in rs) / len(rs)
        lines.append(f"{name:<14}{mean:.3f}±{std:.3f}      {r5:.3f}   {mr:.3f}   {mean - ref:+.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sofuu component ablation runner")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--fast", action="store_true",
                    help="smoke: full,no-moe x 1 seed x 40 steps")
    args = ap.parse_args()

    if args.fast:
        variants, n_seeds, steps = ["full", "no-moe"], 1, 40
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"unknown variants: {unknown} (known: {sorted(VARIANTS)})")
        n_seeds, steps = args.seeds, args.steps

    spec = SynthSpec()
    results = []
    t0 = time.time()
    for variant in variants:
        for s in range(n_seeds):
            r = train_variant(variant, seed=1234 + s, spec=spec, steps=steps)
            results.append(r)
            print(f"[{variant} seed={r['seed']}] R@1={r['recall@1']:.3f} "
                  f"R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f}", flush=True)

    table = summarize(results)
    print("\n" + table)
    print(f"\n({len(results)} runs in {time.time() - t0:.1f}s; "
          "synthetic motif task -- learnability signal only, not production quality)")

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"ablation_{stamp}.json"
    path.write_text(json.dumps({
        "created_utc": stamp,
        "steps": steps,
        "seeds": n_seeds,
        "scope": "synthetic motif retrieval (train motifs != held motifs); "
                 "same data/seeds per variant; CPU proxy L4/d64",
        "results": results,
    }, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
