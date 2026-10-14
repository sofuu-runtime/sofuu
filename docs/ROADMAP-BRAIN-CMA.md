# 🗺️ Roadmap: Brain & CMA — Fixes and Upgrades

> **Status:** proposed 2026-10-13 · milestone-driven, dependency-ordered (no timelines)
> **Scope:** the QTSQ encrypted brain, the Cognitive Memory Architecture, and every surface that
> reads or writes them — CLI, chat, brain HTTP server, agent loop, headless C ABI.
> **Findings source:** `AUDIT-2026-10-13-BRAIN-CMA.md`
> **Companions:** `PLAN-SOFUU-CORE.md` (§3.3 CMA, §3.4 QTSQ), `PLAN-MEMORY-TOKENS.md` (token
> economy), `PLAN-HYBRID-SOFUU-STORAGE.md` (the segment+manifest design), `PLAN-TINY-SEMANTIC-EMBEDDER.md`
> (the two channels).

---

## 0. Ground truth (verified by running, 2026-10-13)

**Working:** brain persistence round-trip (`sofuu doctor` healthy) · decay → retain → consolidate
lifecycle at the turn boundary · near-duplicate reinforcement · entity upsert + agent scoping ·
co-recall gravity · resonance scoring · two-channel hash/sem2 RRF fusion · C2 durability
(fsync → rename → fsync-dir behind a write-ahead journal) · KV page store with 2-page LRU.
`tests/verify_memory_test.js` 30/30; full suite 484/0 in both QTSQ-enabled and QTSQ-free builds.

**Fixed this cycle:** HNSW results-heap ordering — recall@10 vs brute force collapsed 0.82 → 0.24 →
0.07 at n = 50/200/2000 (the results heap shared the candidate frontier's reversed ordering, evicting
near neighbours); now 1.00/1.00/0.91, and end-to-end exact-self top-1 through `sofuu.memory` went
0.70 → 1.00. Two regression tests pin it.

**Open P0:** QTSQ FFI signature drift — `qtsq_compress_tensor` (C: 6 args, Rust: 5) and
`qtsq_compress_tensor_quantized` (C: 7, Rust: 6) read an undefined `level` register on every brain
and KV compression.

**Structural gaps:** no read-out API · metadata-only `/share` + no-op `/import` · dead
`TIER_WORKING` · whole-file brain load/flush (no paging) · fixed shared vault password · no
provenance · no capacity policy · no diversity/recency in scoring · `/forget` unreachable.
*(B1–B5 closed the read-out, share/import, paging, key, provenance, capacity and ranking
items; B6 working memory and B7 harmony remain.)*

---

## 1. Principles

1. **One implementation, four surfaces.** Every memory capability lands in the CMA core, the JS
   binding, the C ABI funnel, and the server in the same milestone — harmony by construction, not
   by follow-up.
2. **Evidence gates.** A milestone is done only when its gate runs green. No ✅ without a run.
3. **Format changes ship with migration + rollback.** The C2 journal + backup discipline is the tool;
   every on-disk change keeps the legacy reader.
4. **No claim ahead of code.** README / EMBEDDING.md / plan text updates ride the milestone that
   makes them true — the "SSD paging" claim is corrected by B3 or not at all.

---

## 2. B0 — Correctness & Safety (first, small)

| # | Item | Size |
|---|---|---|
| B0.1 | ✅ **QTSQ FFI signature fix** (2026-10-13) — `level` added to both tensor compressors; all 7 call sites pass `QTSQ_LEVEL_FULL` (0 = full contest = pre-2.1.0 behaviour). | S |
| B0.2 | ✅ **HNSW recall fix** (2026-10-13) — two regression tests fail pre-fix, pass post-fix. | — |
| B0.3 | ✅ **Codec-regression guard** (2026-10-13) — `codec_stress_flush_reload_120_records`: 120 records through the real save path, reload, count + recall verified. | S |
| B0.4 | ✅ **FFI surface smoke** (2026-10-13) — `ffi_surface_smoke_every_bound_function`: sha256 (NIST vector), container create/add/pack/count/get_stream, both tensor compressors, json + horizon decompress, vault encrypt/is-encrypted/decrypt, write/read — one call each against a real container. | S |

**Gate met (2026-10-13):** full suite **486/0** in both QTSQ-enabled and QTSQ-free builds; the stress
test flushes 120 records and reloads; the smoke test covers the memory subsystem's bound surface
(the session-store functions have their own round-trip test).

---

## 3. B1 — Memory Read-Out, Provenance & Forget (the foundation)

**Objective:** make the brain inspectable. Cards, browsing, editing, capacity reporting and the
memory UI all depend on this one missing capability.

| # | Item | Size |
|---|---|---|
| B1.1 | ✅ **`list` / `export`** (2026-10-13) — `Cma::list_json` (tier/role/entity/text/min-strength filters, offset/limit, filtered `total`; tombstones hidden unless the tier is asked for) + `flat_vectors_for`; JS `cma.list(opts)` / `cma.export()`; `sofuu.memory.inspect(path)` reads a manifest without opening. | M |
| B1.2 | ✅ **Provenance** (2026-10-13) — `created_at`, `last_recalled_at`, `source` on every record; serde-defaulted so legacy files load; wired into pins (`pin`), chat turns (`chat`), the server (`server`), and the C ABI facade. | S |
| B1.3 | ✅ **`/forget`** (2026-10-13) — shows the record + `/forget <id> --yes` to confirm; deletes, flushes, and survives reload (verified). | S |
| B1.4 | ✅ **`/memory` browser** (2026-10-13) — `/memory [tier\|query] [--page N]`, paginated with tier/strength/age/source per row. | M |
| B1.5 | ✅ **Headless + server parity** (2026-10-13) — `memory.list` / `memory.export` in the funnel (anti-drift list now 32); server opens by the file's own manifest and picks the matching embedder; `GET /memories`, `POST /forget`, real `POST /share`; `docs/EMBEDDING.md` table updated. | M |

**Gate met (2026-10-13):** chat e2e (`tests/chat_memory_e2e.sh`, 6/6 on a real pty) lists, confirms,
and deletes; the server smoke drives /remember → /memories → /forget → /share over HTTP; the funnel
test resolves all 32 methods; forget survives reload; legacy records hydrate with defaults; full suite
**497/0** in both builds + capi 44/0.

---

## 4. B2 — Real Brain Cards (share / import)

**Objective:** replace the metadata-only stubs with working portable cards.

| # | Item | Size |
|---|---|---|
| B2.1 | ✅ **`/share` writes a real card** (2026-10-13) — `sofuu-brain-card@2`: records + vectors + manifest + label, via `cma.export()`. | M |
| B2.2 | ✅ **`/import` merges** (2026-10-13) — `Cma::import_records` + `cma.importRecords(vecs, recordsJson)`: same-space cards reuse their vectors, cross-space cards re-embed every text locally, near-duplicates reinforce (max strength, earliest `created_at`, original text/role/source kept), tombstones never import, and the merge is idempotent. Also fixed: `fs.readFile/writeFile` are promise-based and the pre-B2 `/import` never awaited them, so it always took the "not a text file" branch. | M |
| B2.3 | ✅ **Server endpoints** (2026-10-13) — `POST /share` returns the real card; `GET /memories` + `POST /forget` on the same read-out API. | S |
| B2.4 | ✅ **Card format spec** (2026-10-13) — `docs/BRAIN-CARD.md`: shape, the records↔vectors pairing rule, spaces, import semantics, versioning (v1 refused by name). | S |

**Gate met (2026-10-13):** live round-trip — export 2 records → import into a fresh brain (added 2)
→ import again (merged 2, count unchanged) → reload keeps them with provenance; the chat e2e drives
`/share` in one project and `/import` + `/memory` in a second (9/9 on a real pty); the server smoke
covers `POST /share`. Full suite **500/0** in both builds.

---

## 5. B3 — Brain Storage: Segments, Capacity, Compaction

**Objective:** stop rewriting the whole brain on every flush, and bound its growth.
Measured baseline: 2000 records → 6.2 MB file, 195 ms flush, 311 ms reopen — all O(total).

| # | Item | Size |
|---|---|---|
| B3.1 | ✅ **Segmented layout** (2026-10-13) — `rt/brain_store.rs`: `<path>` is a manifest container, records live in `<path>.d/seg-<epoch>-<seq>.qtsq` (tensor + meta streams). One committed manifest, immutable sealed segments, at most one mutable active segment — the `session_store.rs` contract, behind the C2 journal. | L |
| B3.2 | ✅ **Incremental flush** (2026-10-13) — the CMA tracks dirty ids (recall reinforcement, tier changes, tombstones, upserts) and the flush appends exactly those plus never-persisted records; the active segment is rewritten only while it has room. Measured live: 5 new records cost **548 bytes** where the old full rewrite cost ~80 KB. Decay is not written at all — the manifest carries `last_decay_at` and a load replays the elapsed decay (capped at 7 days, the same bound the turn tick uses). Legacy single-file brains still load. | M |
| B3.3 | ✅ **Capacity policy** (2026-10-13) — `SOFUU_BRAIN_MAX_RECORDS` (default 100k, 0 disables). Over the cap the weakest live records are tombstoned (never pins, never entities; oldest first among equals) and the count is reported in `cma.stats().evicted_total`. Tombstoning rather than removing keeps ids stable, so a caller holding recall ids across the flush cannot be handed a wrong record; `retain()` drops them at the next turn boundary. | M |
| B3.4 | ✅ **Compaction** (2026-10-13) — automatic when superseded versions exceed 2× live + 64, or when `retain()` renumbered / `cma.compact()` was called; rewrites one version per live record into a fresh epoch and preserves recall (verified). Old generations stay on disk until the age-guarded orphan sweep (10 min) can prove no concurrent reader holds them. | M |
| B3.5 | ✅ **Channel lifecycle dedup** (2026-10-14) — the SEM2 sibling is a DERIVED index over the canonical records: it no longer runs its own decay or retain. At the turn boundary, once the canonical store has pruned (or consolidated), the sibling is pruned to the same live set by text via the new cross-store `cma.retainAgainst(canonical)` (aliasing-guarded; renumbers exactly like `retain()`). One decay, one prune — the two channels can no longer disagree about which memories exist. **Honest scope:** the two VECTORS cannot be deduped (a 768-dim hash vector and a 64-dim semantic vector are different spaces and both are needed for the two search channels); what is deduped is the lifecycle — the duplicated per-turn decay/retain work and the drift it produced. | M |

**B3.5 gate met (2026-10-14):** `retain_against_prunes_the_sibling_to_the_canonical_set` (canonical
prunes 2 → sibling prunes 2 → both hold the same live set; a synced sibling prunes nothing; a store
against itself is a guarded no-op) plus `retain_matching_follows_the_canonical_set` /
`retain_matching_remaps_corecall_partners` at the core, and a live probe through real agent turns: a
record forgotten in the canonical store is still present in the sibling before the turn boundary and
gone after it, with both live sets identical. Full suite **526/0** in both builds; the agent battery
and the 30/30 memory suite stay green.

**Gate met (2026-10-13):** `incremental_flush_touches_only_the_active_segment` (sealed segments
byte-identical, only the active one + manifest change); `ten_thousand_records_reload_and_recall`
(10k records, reload, recall); `capacity_evicts_weakest_and_reports` (pins and entities survive);
`compaction_preserves_recall_and_shrinks_versions`; `legacy_brain_migrates_with_backup_and_keeps_records`;
`torn_segment_is_refused`. Full suite **512/0** in both builds. Live probe: 600 records → 3 segments,
605 records → +548 bytes, reload + recall + compact all correct.

**Found while testing (codec-level, reported not fixed):** the QTSQ container read accepts a
TRUNCATED file as a valid, silently-shortened container — a segment cut to 1/3 loaded as 6 of 10
records with no error. The store therefore carries a per-segment SHA-256 in the manifest and refuses
a mismatch, so a torn write can never read as a brain that quietly forgot things.

---

## 6. B4 — Encryption Hardening

**Objective:** replace the fixed shared vault password (`"sofuu-kv-local-v1"`, identical on every
machine) with a machine-local key.

| # | Item | Size |
|---|---|---|
| B4.1 | ✅ **Key derivation** (2026-10-14) — `rt/brain_key.rs`: `SOFUU_BRAIN_KEY` (64 hex) → `~/.sofuu/brain.key` (32 random bytes, `0600`, written to a private temp sibling and renamed, cached per process). The codec's device keystore was evaluated first and rejected — macOS Keychain prompts unsigned binaries on every launch, and its ML-KEM path is a different on-disk scheme; the reasoning is recorded in `BRAIN-KEY.md`. | M |
| B4.2 | ✅ **Migration** (2026-10-14) — reads try the machine key then the legacy password (the codec verifies the HMAC before touching the ciphertext, so the retry is free). A legacy-opened brain is marked, backed up to `<path>.legacy-key.bak` on its first write (refusing the write if the backup fails), and re-encrypted through the compaction path. | M |
| B4.3 | ✅ **Recovery story** (2026-10-14) — `docs/BRAIN-KEY.md`: resolution order, migration, and a situation/action table for a new machine, a lost key file, and sharing. Says plainly that the brain is machine-local by design and that a lost key with no backup is unrecoverable. | S |

**Gate met (2026-10-14):** `fresh_brain_uses_the_machine_key` (the legacy password cannot open a new
brain), `legacy_encrypted_brain_migrates_with_backup` (opens, backs up, re-encrypts, recalls, and the
legacy password stops working), `wrong_key_fails_closed_without_corrupting` (refuses, file
byte-identical, the right key still opens). Also fixed while testing: a wrong-key **segmented** brain
used to fall through to the single-file reader and open as EMPTY — `read_manifest_ex` now separates
"not a container" (the corrupt-file backup path) from "a container this key cannot open" (refuse).
Full suite **532/0** in both builds.

---

## 7. B5 — Recall Quality

**Objective:** better ranking on real brains, measured rather than asserted.

| # | Item | Size |
|---|---|---|
| B5.4 | ✅ **Recall evaluation harness** (2026-10-14) — `memory::recall_eval`: a deterministic fixture (one chatty topic of five paraphrases, two sparse topics, 40 noise records, dim 256) + a query touching all three, reporting recall@k, MRR, topic coverage@k and redundancy@k; plus an ef sweep against brute force and a recency tie-break. Run with `cargo test -p sofuu-core --lib recall_eval -- --nocapture`. | M |
| B5.1 | ✅ **Diversity** (2026-10-14) — near-duplicate suppression while filling top-k (`diversify`, floor 0.6 cosine). The roadmap said MMR; the harness measured the textbook linear trade-off promoting *unrelated* records on this scoring scale (relevance 1.00 → 0.17 with noise pulled above the secondary topics), so the shipped mechanism is MMR-family diversity by suppression: relevance order is never inverted, near-duplicates are skipped. | S |
| B5.2 | ✅ **Recency term** (2026-10-14) — `+0.15 · exp(-age/7d)` on the composite score from B1.2's `created_at`; fresh records outrank equally-strong stale ones, `created_at == 0` (legacy) stays neutral. | S |
| B5.3 | ✅ **`ef_search` knob** (2026-10-14) — `Hnsw::search_with_ef` + `Cma::set_ef_search`, read from `SOFUU_BRAIN_EF_SEARCH` on every surface that opens a brain (JS shell, C ABI facade), reported in `cma.stats()`. Default unchanged. | S |

**Measured (harness, k=6):** baseline (diversity off) recall@6 **1.000**, MRR **1.000**, topic
coverage **0.667**, redundancy **0.667** → shipped defaults recall@6 **0.500**, MRR **1.000**,
coverage **1.000**, redundancy **0.500** — the top-k stops returning five paraphrases of one fact
and touches all three topics, at the cost of two slots that held redundant copies. The floor sweep
is flat across 0.4–0.7 (0.80 behaves like off), so 0.6 sits mid-plateau. ef sweep (n=600, k=10):
ef 8 → 0.685, default 40 → 0.975, ef 512 → 1.000 against brute force.

**Gate met (2026-10-14):** harness numbers above; full suite **543/0** in both QTSQ-enabled and
QTSQ-free builds; `tests/verify_memory_test.js` **30/30** on a real QTSQ binary.

---

## 8. B6 — Working Memory (the fourth tier, real)

**Objective:** make `TIER_WORKING` real without polluting the persistent index. Recommended shape:
an in-RAM working set held **beside** the persistent record table (not inside it), merged into
recall results with the existing 0.5 tier weight — this avoids index renumbering entirely.

| # | Item | Size |
|---|---|---|
| B6.1 | ✅ **In-RAM working set** (2026-10-14) — `Cma::working` beside the persistent table: never in `records`, never in the HNSW index, never serialized, capped at 256 (oldest evicted). Recall merges it with the 0.5 working tier weight + the B5.2 recency boost and no resonance history; the B5.1 diversity floor suppresses a working twin of a persistent fact. Ids carry bit 30 (positive through i32/JSON; persistent ids can never reach 2^30), so every consumer can tell which tier served a hit. | M |
| B6.2 | ✅ **Promotion** (2026-10-14) — at the turn commit, the turn's recall-hit ids (channel-split exactly like markPositive: canonical ids → the canonical store, sem-only ids → the sem store) promote via `promoteWorking` into the persistent store as episodic. A near-duplicate of a persistent record reinforces it (merged) instead of duplicating; the promoted record keeps its ORIGINAL `created_at` and source. Turn facts (task/answer) now land in the working set first — a fact no turn ever recalls dies with the session. | M |
| B6.3 | ✅ **Chat integration** (2026-10-14) — current-session facts are recallable immediately; `/why` labels each hit with the tier that served it (`[working]` highlighted, `[episodic]`/`[semantic]`/`[entity]` dim). JS surface: `rememberWorking` / `promoteWorking` / `clearWorking` / `stats().working`; C ABI: `sofuu_cma_remember_working` / `sofuu_cma_promote_working`. An older binary without the bindings falls back to direct persistent writes. | S |

**Gate met (2026-10-14):** working facts recall within a session (16/16 JS gate test: immediate
recall on an empty store; flush + reload shows none; promotion reports `promoted`/`merged`, moves
the record to episodic with the original write time preserved, merges a persistent duplicate;
`clearWorking` drops unpromoted facts); full suite **549/0** in both QTSQ-enabled and QTSQ-free
builds; `tests/verify_memory_test.js` 30/30; `tests/chat_memory_e2e.sh` 9/9 on a real pty
(mock LLM, no network).

---

## 9. B7 — Harmony & Surface Coherence (last)

**Objective:** the brain behaves the same everywhere and says what it does.

| # | Item | Size |
|---|---|---|
| B7.1 | **One semantics across surfaces** — same filters, same error codes, same tier names in CLI, chat, server and C ABI; pinned by a cross-surface test. | M |
| B7.2 | **`sofuu doctor` brain section** — manifest/space, record + tier histogram, capacity headroom, journal state, last compaction. | S |
| B7.3 | **Chat brain panel** — `/brain info` shows tiers, capacity, channels, last consolidation. | S |
| B7.4 | **Docs truth pass** — README, `EMBEDDING.md`, `PLAN-SOFUU-CORE.md` reflect shipped behaviour only. | S |

**Gate:** cross-surface test green; doctor output on a real brain shows real numbers; no doc claim
without a passing test behind it.

---

## 10. Dependency order

```
B0 (correctness)  ──────────────────────────► everything
B1 (read-out + provenance) ──► B2 (cards)
                           ├──► B5.2 (recency)
                           └──► B6.3 (tier display)
B3 (storage) ──► B4 (key migration rides B3's rewrite if done back-to-back)
B5 (quality) — independent of B3/B4; needs B1.2
B6 (working tier) — independent; display needs B1
B7 (harmony) — last: it pins the shipped behaviour
```

B3 and B4 both touch the persistence path; doing B3 first means the key migration is written once
against the final layout. If security is the priority, B4 can be pulled ahead of B3 — the cost is a
second migration pass.

---

## 11. Out of scope

- Changing the QTSQ **container** format itself (adapter-level layout only; the codec stays as-is).
- Vector quantization for the brain (the codec compresses; precision is a codec concern).
- Embedding-model changes (separate: `PLAN-TINY-SEMANTIC-EMBEDDER.md`, emb-m / C3).
- ML-gate shadow A/B telemetry (separate: `PLAN-SOFUU-CORE.md` C4).
- Hosted / sync / multi-device services.

---

## 12. Suggested first three pulls

1. **B0.1 + B0.3** — smallest, and it removes the undefined-behaviour hazard on every compression.
2. **B1.1 + B1.2** — read-out + provenance; unlocks every visible feature downstream.
3. **B2** — turns the README's brain-card promise true.
