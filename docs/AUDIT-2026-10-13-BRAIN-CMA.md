# AUDIT — 2026-10-13 — QTSQ Encrypted Brain & Cognitive Memory Architecture

> Scope: `crates/sofuu-core/src/memory/{cma,hnsw}.rs`,
> `crates/sofuu-core/src/rt/memory.rs` (the `sofuu.memory` / `sofuu.kv` /
> `sofuu.agent` JS surface), the chat brain commands, the brain HTTP server,
> and the QTSQ FFI bindings. Verified by running, not by reading claims.
>
> Companion plans: `PLAN-SOFUU-CORE.md` §3.3/§3.4 (canonical), `PLAN-MEMORY-TOKENS.md`,
> `PLAN-HYBRID-SOFUU-STORAGE.md`.

## 1. What was verified working (live, 2026-10-13)

| Area | Evidence |
|---|---|
| Brain persistence round-trip | `sofuu doctor` → healthy; QTSQ linked, write→read round-trip passes |
| CMA mechanics | `tests/verify_memory_test.js` **30/30** — dedup/reinforce, Ebbinghaus decay, tier promotion, GMF co-recall, dream consolidation, resonance, reload persistence |
| Tier promotion | episodic → semantic (consolidation), episodic → entity (upsert), forgotten tombstones + `retain()` physical prune |
| Decay + prune at turn boundary | `brainDecayTick` in `src/js/agent.js` (decay → retain → consolidate-if-due) |
| Two-channel recall fusion | hash-768 + sem2-64 RRF fusion with per-channel markPositive split |
| Durability (C2, 2026-10-12) | fsync→rename→fsync-dir publish + write-ahead journal; recovery on open |
| KV page store | 2-page LRU, quantized K/V streams, index.json staged + journalled |

## 2. Defects found

### 2.1 [FIXED 2026-10-13] HNSW recall collapsed as the brain grew — P0

`Hnsw::search_layer` used ONE candidate type (`Cand`) with a reversed `Ord`
for both heaps, so `results` was a min-heap used as a max-heap: it evicted the
CLOSEST neighbours and the early-exit bound compared against the BEST distance
instead of the worst.

Measured (recall@10 vs brute force, 100 queries, dim 64):

| n | before | after |
|---|---|---|
| 50 | 0.820 | 1.000 |
| 200 | 0.236 | 1.000 |
| 2000 | 0.072 | 0.910 |

End-to-end through the real `sofuu.memory` API with 300 stored memories,
querying with the EXACT stored text: top-1 self-hit **0.70 → 1.00**,
top-5 self-hit **0.88 → 1.00**.

The bug was invisible at the sizes the test suite used (ef ≥ n never trims),
so every existing test passed. Fixed by splitting the results heap into its own
naturally-ordered type; pinned by two regression tests
(`recall_stays_high_as_the_graph_grows`, `exact_self_is_always_top1`) that fail
pre-fix (0.230) and pass post-fix.

### 2.2 [OPEN — other agent's in-flight file] QTSQ FFI signature drift — P0

The codec's 2.1.0 API added a trailing `int level` parameter. The C header
(`qtsq.h`) now declares:

```c
int qtsq_compress_tensor(ctx, data, count, dims, num_dims, level);                    /* 6 */
int qtsq_compress_tensor_quantized(ctx, data, count, dims, num_dims, precision, level); /* 7 */
```

`crates/sofuu-ffi/src/qtsq.rs` still declares 5 and 6 parameters. Calling a
6/7-argument C function through a 5/6-argument declaration is undefined
behaviour: the callee reads an undefined register as `level`. The build-time
`_Static_assert` guard covers the STRUCT LAYOUT only — it cannot see function
signatures, which is why this slipped through.

Impact: every brain memory-tensor compression and every KV page compression
runs with an undefined `level`. Fix: add the parameter and pass the intended
level explicitly (0 = full contest preserves the pre-2.1.0 behaviour).

### 2.3 [TRANSIENT — fixed upstream] Codec heap corruption on brain flush

With the codec build of 2026-10-13 10:14, `brain.flush()` aborted (SIGABRT) at
≥ ~17-20 records inside `qtc_compress` ← `qtsq_compress_json` ← `cma_flush_to`
("freeing a pointer that was not allocated"). The codec owner rebuilt
`libqtc.a`/`libqtsq.a` at 10:51/10:53 and the crash disappeared (re-verified:
flush of 2000 records succeeds). Recorded here because it is the class of
failure — codec-side, surfacing in the brain path — that the brain's own tests
cannot catch.

## 3. Structural gaps vs the plan and README

| Gap | Detail |
|---|---|
| **No read-out API** | `sofuu.memory` exposes remember/recall/forget/count/… but NOTHING that enumerates records. `records_json` exists internally (it is the file metadata) yet is not exposed. Every "browse / export / edit / share" feature is blocked on this one omission. |
| **`/share` + `/import` are stubs** | `/share` writes a metadata-only JSON card (schema + count + path — no vectors, no text). `/import` reads the card and prints "Full vector-level merge requires the source brain file"; it merges nothing. The brain HTTP server's `POST /share` returns `"share endpoint not yet implemented"`. The README advertises "portable brain cards". |
| **Working-memory tier is dead** | `TIER_WORKING` is defined and weighted in scoring but never assigned — every record is born `TIER_EPISODIC`. The advertised 4-tier model is 3-tier in practice. |
| **No SSD paging for the brain** | The plan claims "2-page RAM LRU with direct SSD paging … search gigabyte-sized stores from 512 MB RAM". The KV store has the LRU; the BRAIN loads the whole file into RAM and rewrites the whole file on every flush. Measured: 2000 records → 6.2 MB file, flush 195 ms, reopen 311 ms. A 1 GB brain would need ~1 GB RAM and ~2 s per flush. |
| **Encryption is a fixed password** | `SOFUU_QTSQ_LOCAL_PASSWORD = "sofuu-kv-local-v1"` — identical on every machine. The code comments call it "at-rest obfuscation"; the plan says "machine-local key". No per-machine key derivation exists. |
| **No `/forget` and no memory editing** | `forget(id)` is exposed to JS but never called by any command or driver path. There is no way for a user to see or delete a specific memory. |
| **No memory provenance** | Records carry `role` + `age_seconds`; there is no created-at, no source (file/turn/tool), no session id. "When/where did I learn this" is unanswerable. |
| **No write-time cross-channel dedup** | The same text is stored in BOTH the hash store and the sem2 store; fusion dedups at read time by exact text match. Storage and decay are paid twice. |
| **Brain server hardcodes the space** | `serve --brain` opens dim 768 and embeds with `embedLocal` (hash-768) regardless of the brain file's actual embedding manifest. |
| **No capacity policy** | No max-records / max-bytes cap, no eviction beyond decay+retain, no compaction pass. |
| **Scoring has no recency or diversity term** | Score = (1 − normalized distance) × strength × tier × resonance. No MMR/diversity (top-k can be near-duplicates), no explicit recency, no query expansion. |

## 4. Suggested upgrade order

1. **P0 — FFI signature fix** (§2.2) — owner of the in-flight FFI migration.
2. **P0 — read-out API** (`list`/`export` with tier/role/offset filters) — unblocks share/import, memory browsing, editing, and the UI.
3. **P1 — real brain cards**: `/share` exports vectors + records; `/import` merges with re-embedding + dedup; server `POST /share` implemented on top of the read-out API.
4. **P1 — working-memory tier**: session-scoped records that never persist, promoted to episodic on turn commit.
5. **P1 — provenance**: created_at + source on every record (schema-compatible: new optional fields).
6. **P2 — segmentation/paging**: bounded segments + manifest (the design already written in `PLAN-HYBRID-SOFUU-STORAGE.md`) applied to the brain, so flush stops rewriting everything.
7. **P2 — per-machine key**: derive the vault password from a machine-local secret instead of the shared constant.
8. **P2 — recall quality**: diversity (MMR) + recency term + ef_search tuning knob.
9. **P3 — `/forget` + memory browser** in chat, on top of the read-out API.
