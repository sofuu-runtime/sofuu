# Brain Cards — the portable memory format

> **Status:** `sofuu-brain-card@2` shipped 2026-10-13 (ROADMAP-BRAIN-CMA B2).
> Produced by `/share` (chat), `POST /share` (brain server) and
> `cma.export()` (JS / C ABI). Consumed by `/import` (chat) and
> `cma.importRecords()`.

A brain card is one JSON document: the records a brain holds, the vectors
those records are indexed by, and the manifest that says which embedding
space the vectors belong to. It is deliberately plain — no QTSQ container,
no encryption — so a card can be read, diffed and inspected anywhere.

## Shape

```jsonc
{
  "schema": "sofuu-brain-card@2",
  "exported_at": 1791879023,          // unix seconds
  "label": "my project brain",        // optional, set by /share <path> <label>
  "embedding": {
    "id": "hash-v1",                  // the vector space (see Spaces below)
    "input": "hash-v1",
    "dimension": 768,
    "artifact": ""                    // model artifact id for learned spaces
  },
  "total": 2,                         // == records.length == vectors.length / dimension
  "records": [                        // the read-out shape (cma.list)
    {
      "id": 0,
      "tier": 1,                      // 0 working · 1 episodic · 2 semantic · 3 forgotten · 4 entity
      "strength": 1.0,
      "role": "user_pin",
      "text": "the deploy bucket is sofuu-dist",
      "entity": null,                 // entity namespace, for tier 4
      "entity_type": null,
      "created_at": 1791879023,       // 0 = unknown (legacy)
      "last_recalled_at": 0,          // 0 = never
      "source": "pin",                // where it came from
      "age_seconds": 0,
      "total_recalls": 0,
      "positive_recalls": 0,
      "kv_page_id": 0
    }
  ],
  "vectors": [0.012, -0.4, …]         // row-major, records[i] ↔ vectors[i*dim .. (i+1)*dim]
}
```

**The pairing rule is the contract:** `records[i]` is indexed by
`vectors[i * dimension .. (i + 1) * dimension]`. An importer must refuse a
card whose `vectors.length != records.length * dimension` when the spaces
match — a misaligned card would silently attach the wrong vector to every
record.

Tombstones (tier 3) are never exported and never imported: a card is a
snapshot of what the brain holds, not of what it deleted.

## Spaces

| `embedding.id` | dimension | produced by |
|---|---|---|
| `hash-v1` | 768 | `sofuu.ai.embedLocal` (default, zero-byte) |
| `semantic-projector-v1` | 64 | `sofuu.ai.embedLocalSemantic` |
| `semantic-table-v2` | 64 | `sofuu.ai.embedLocalSemanticV2` |
| `provider:model@rev` | any | a remote embedding provider |

## Import semantics

1. **Same space** (`card.embedding.id == local space` and dimensions equal):
   the card's vectors are used as-is.
2. **Different space**: every `records[i].text` is re-embedded locally with
   the embedder for the LOCAL space; the card's vectors are discarded. If no
   local embedder exists for the local space, the import refuses.
3. **Merge policy** per record: a near-duplicate of an existing live record
   (vector distance below the store's duplicate gate) reinforces it — the
   higher strength wins, the earliest `created_at` wins, and the original
   text/role/source are kept. Everything else inserts with the card's tier,
   role, entity and provenance. `source` becomes `"import"` when the card
   does not carry one.
4. **Idempotent**: importing the same card twice adds nothing the second
   time — every record merges.
5. The result is `{added, merged, skipped}`.

## Versioning

- `sofuu-brain-card@1` (2026-09, retired): metadata only — schema, count,
  path. No records, no vectors. `/import` refuses it with a clear message
  rather than pretending to merge.
- `sofuu-brain-card@2` (current): the shape above. Unknown versions are
  refused by name.

## Size

JSON carries vectors as decimal numbers, so a card is roughly
`records × dimension × ~12 bytes`. A 2,000-record hash-768 brain is about
18 MB. That is fine for sharing a project brain; a much larger store should
be shared as the `.qtsq` file itself (the codec compresses and encrypts it).
