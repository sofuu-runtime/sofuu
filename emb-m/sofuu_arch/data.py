"""Training data: JSONL pairs + synthetic smoke data (original code).

Real-data format (one JSON object per line):

    {"query": [12, 44, 9, ...],
     "positive": [7, 7, 31, ...],
     "hard_negatives": [[...], [...]]}     # optional, 0+ id lists
    {"query": [...], "positive": [...]}    # hard negatives optional

Ids are pre-tokenized integer sequences (the BPE trainer plugs in here and
nothing else changes). Sequences longer than max_seq_len are truncated;
shorter ones are right-padded with pad_id (0) and masked out.

For 2xT4 smoke tests without real data, motif pairs from synth.py are
wrapped into the same tensor format.
"""

import json

import torch
from torch.utils.data import Dataset

from .synth import SynthSpec, train_pairs


class PairDataset(Dataset):
    """(query_ids, doc_ids, doc_mask) with optional packed hard negatives.

    Each item returns:
        q_ids   (L,) long      padded query
        q_mask  (L,) bool      True == real token
        d_ids   (1+H, L) long  row 0 = positive, rows 1..H = hard negatives
        d_mask  (1+H, L) bool
    H varies per record; the collate pads H to the batch max.
    """

    def __init__(self, records: list, max_seq_len: int, pad_id: int = 0):
        self.records = records
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.records)

    def _pack(self, ids: list) -> tuple[torch.Tensor, torch.Tensor]:
        ids = ids[: self.max_seq_len]
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        out = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[len(ids) :] = False
        return out, mask

    def __getitem__(self, i: int):
        r = self.records[i]
        q_ids, q_mask = self._pack(r["query"])
        docs = [r["positive"]] + list(r.get("hard_negatives", []))
        packed = [self._pack(d) for d in docs]
        d_ids = torch.stack([p[0] for p in packed])
        d_mask = torch.stack([p[1] for p in packed])
        return q_ids, q_mask, d_ids, d_mask


def collate_pairs(batch: list) -> tuple:
    q_ids = torch.stack([b[0] for b in batch])
    q_mask = torch.stack([b[1] for b in batch])
    max_h = max(b[2].shape[0] for b in batch)
    L = batch[0][2].shape[1]
    d_ids = torch.zeros(len(batch), max_h, L, dtype=torch.long)
    d_mask = torch.zeros(len(batch), max_h, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        h = b[2].shape[0]
        d_ids[i, :h] = b[2]
        d_mask[i, :h] = b[3]
    return q_ids, q_mask, d_ids, d_mask


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "query" not in r or "positive" not in r:
                raise ValueError(f"{path}:{line_no}: record needs query + positive")
            records.append(r)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def motif_records(n: int, seed: int, spec: SynthSpec | None = None) -> list:
    """Smoke-test records from the synthetic motif task (same format)."""
    spec = spec or SynthSpec()
    q, d, _ = train_pairs(spec, n, seed)
    return [
        {"query": q[i].tolist(), "positive": d[i].tolist()} for i in range(n)
    ]
