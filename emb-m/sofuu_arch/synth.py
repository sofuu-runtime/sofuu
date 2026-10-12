"""Synthetic motif-retrieval task for component ablations (original code).

Each "motif" is a short fixed token pattern. A query and its positive
document are independent noisy renders of the SAME motif (pattern placed at
a random offset inside uniform noise); negatives are renders of OTHER
motifs. Relevance is therefore unambiguous and learnable: the model must
match separated placements of one pattern while ignoring noise.

Train motifs and held-out motifs are DISJOINT, so evaluation measures
generalization to unseen patterns, not memorization of training renders.

Honest scope: this tests whether a component helps a transformer LEARN a
clean retrieval regularity on CPU in minutes. It does NOT predict real
code/doc retrieval quality. A component winning here earns a pilot-scale
run on real data; nothing here earns a production claim.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SynthSpec:
    vocab_size: int = 512
    seq_len: int = 32
    motif_len: int = 6
    n_train_motifs: int = 24
    n_held_motifs: int = 12
    # Motif tokens come from a small alphabet, so motifs OVERLAP heavily.
    # Bag-of-words / content-only models cannot separate them; the model
    # must use token ORDER (position) and sharper features to win.
    motif_alphabet: int = 16


def make_motifs(rng: torch.Generator, n: int, spec: SynthSpec) -> torch.Tensor:
    return torch.randint(0, spec.motif_alphabet, (n, spec.motif_len), generator=rng)


def render(rng: torch.Generator, motif: torch.Tensor, spec: SynthSpec) -> torch.Tensor:
    ids = torch.randint(0, spec.vocab_size, (spec.seq_len,), generator=rng)
    max_off = spec.seq_len - spec.motif_len
    off = int(torch.randint(0, max_off + 1, (1,), generator=rng).item())
    ids[off : off + spec.motif_len] = motif
    return ids


def train_pairs(
    spec: SynthSpec, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (queries, docs, motif_ids): pair i shares motif motif_ids[i]."""
    rng = torch.Generator().manual_seed(seed)
    motifs = make_motifs(rng, spec.n_train_motifs, spec)
    qs, ds, ms = [], [], []
    for _ in range(n):
        m = int(torch.randint(0, len(motifs), (1,), generator=rng).item())
        qq, dd = _render_pair(rng, motifs, m, spec)
        qs.append(qq)
        ds.append(dd)
        ms.append(m)
    return torch.stack(qs), torch.stack(ds), torch.tensor(ms)


def _render_pair(rng, motifs, m, spec):
    return render(rng, motifs[m], spec), render(rng, motifs[m], spec)


def eval_set(
    spec: SynthSpec, n_queries: int, pool_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (queries [Q,L], pool [P,L], query_motifs [Q], pool_motifs [P]).

    Relevance is motif-level: ANY pool doc sharing the query's motif is a
    correct retrieval. Metrics must use motif match, never exact-index match.
    """
    rng = torch.Generator().manual_seed(seed)
    motifs = make_motifs(rng, spec.n_held_motifs, spec)
    queries, positives, q_motifs = [], [], []
    for _ in range(n_queries):
        m = int(torch.randint(0, len(motifs), (1,), generator=rng).item())
        qq, dd = _render_pair(rng, motifs, m, spec)
        queries.append(qq)
        positives.append(dd)
        q_motifs.append(m)
    queries = torch.stack(queries)
    positives = torch.stack(positives)
    q_motifs = torch.tensor(q_motifs)
    n_dist = pool_size - n_queries
    dist, dist_m = [], []
    for _ in range(n_dist):
        m = int(torch.randint(0, len(motifs), (1,), generator=rng).item())
        dist.append(render(rng, motifs[m], spec))
        dist_m.append(m)
    pool = torch.cat([positives, torch.stack(dist)], dim=0)
    pool_motifs = torch.cat([q_motifs, torch.tensor(dist_m)])
    perm = torch.randperm(len(pool), generator=rng)
    return queries, pool[perm], q_motifs, pool_motifs[perm]
