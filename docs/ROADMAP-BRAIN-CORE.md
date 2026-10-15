# ROADMAP-BRAIN-CORE — the precision-first memory program (Brain Core II)

> **Status:** M0–M3 landed and pushed 2026-10-15. Companion to
> `ROADMAP-BRAIN-CMA` (B0–B7: tiers, decay, QTSQ stores, cards, HNSW) —
> that program built the brain; this one made it precise. The charter,
> verbatim from the developer: *"it's not about how many features we add,
> it's about the features we add working with insane precision and rarely
> failing their job — this applies to every single feature and function."*
> Every section below carries its measured gate; nothing is claimed ahead
> of code.

## The program

| Milestone | Focus | Landed in |
|---|---|---|
| M0 | The precision floor: invariant fuzz + long-session soak + the agent-memory collapse fix | `c457f2896` |
| M1.1 | Live-window recall budget | `5da80d95c` |
| M1.2 | MMR backfill + recall targets | `4d3d80dca` |
| M2 | Reinforcement buys decay tolerance + the DeathLedger | `7248d7607` |
| M3 | `/compact` lands the distilled gist | `4351a00bf` |

## M0 — the fuzz and the soak (the gates every later change passed through)

`memory/invariant_test.rs` — model-based fuzz, 3 seeds × 12k adversarial
ops against a shadow model, asserting the product contract: dense ids, no
indirect resurrection of forgotten texts, self-recall of live rows,
flush→hydrate fidelity, working cap, promotion provenance, mark-positive
purity, pin/entity survival, no tombstone leaks in the default view.

It earned its keep on day one — three real defects, none reachable by the
existing unit tests:

| Defect | Mechanism | Fix |
|---|---|---|
| Consolidation destroyed pins | `consolidate()`'s weak-episodic sweep had no pin guard: an aged, weak `user_pin` was folded into a dream cluster and tombstoned | Pin-role exclusion in the sweep; regression test `consolidate_never_tombstones_a_pin` |
| Entity insert kidnapped rows | `remember_entity`'s insert arm stamped entity identity (tier 4 + name) onto whatever row the dup gate MERGED into | Identity lands only on genuinely fresh inserts (the consolidate guard, applied twice) |
| Namespace relabel across agents | `remember_entity_scoped` had the same shape — a merged row was re-labeled into the CALLER's namespace, a privacy-boundary crossing | Same fresh-only stamp |

The day-0 soak (`tests/memory_soak_e2e.sh`, gitignored `/tests/` — JS
gates never commit) found the collapse that started all of this: agent
private memory stored via `rememberEntity` **upserts by name** — 200 turns
kept exactly 1 row. The fix is `rememberEntityTurn` + `remember_entity_scoped`
(entity-named rows that rank and decay as EPISODIC, stamped only on fresh
inserts) plus per-turn nested namespace keys in `agent.js`. Measured: 200
turns → 220 rows where there was 1; **20/20 canaries recall across a
process restart, 0 cross-canary contamination**.

One ranking defect surfaced under decay churn and is now the reflexive
recall law: a query vector that *is* a stored record's question outranks
every inexact hit regardless of decay (REFLEX_EPS 0.02 bucket).

## M1 — recall budgeting

### Live-window budget (`5da80d95c`)

The recall block now answers to the tokens actually free in the upcoming
request. `refreshPlan()` measures: resolved window − (system + tools) −
history − task − attachments − answer reserve. `allocRecallBudget()`
ceilings the plan's shaped budget with that measurement; below
`RECALL_MIN_FREE_TK` (768 tk) recall injects nothing and says so
(`recall {skipped:'window', budgetCut}`). Budget 0 is authoritative — an
unset def carries `undefined`, never 0 (the old first-line allowance made
"zero budget" mean "one line"). Under real pressure memory *yields to the
conversation* instead of making the fitGuard ladder trim real history to
make room for it.

### Ranking targets (`4d3d80dca`)

`diversify()` fills every slot: after the near-duplicate suppression
pass, the remaining slots take suppressed candidates in relevance order
(the B5.1 contraction returned fewer than top_k and starved recall).
Measured on `memory/recall_eval` (7 topics [6,2,2,2,1,1,1] + 40 noise,
k=6, floor 0.60):

| Config | recall@6 | MRR | coverage | redundancy |
|---|---|---|---|---|
| floor off (pre-B5 order) | 1.000 | 1.000 | 0.143 | 0.833 |
| B5.1 contraction (before M1.2) | 0.500 | 1.000 | 1.000 | 0.500 |
| **M1.2 default (backfill)** | **1.000** | **1.000** | 0.857 | **0.000** |

Asserted bar: recall@6 ≥ 0.95, MRR ≥ 0.99, redundancy ≤ 0.20, coverage
and redundancy must beat the no-floor baseline. (The fixture grew from 3
topics to 7 — with three topics, redundancy@6 can never read below 0.5
regardless of selection quality, so the metric was physically
unassertable.)

## M2 — forgetting, properly designed

### Reinforcement buys decay tolerance (`7248d7607`)

The untouched 1-day curve killed memories *between their own uses*: a
memory touched every week read exp(−7) ≈ 0.0009 by day 7 — below the
0.05 floor, so retain pruned it **before its next use**. The law now:

> each positive recall at least **doubles** the half-life (never less
> than +2 days — the SM-2 insight that interval growth, not a fixed
> multiplier, keeps a cadence alive), capped at the tier curve — episodic
> 30 d, semantic 90 d. Entities (1 year) need no help.

`mark_positive` still never moves strength (the I7 contract); `kv_hints`
' growth rides the same caps (the old ×1.2 there was unbounded — a hot KV
row could reach a thousand-day half-life). Measured in
`memory/decay_eval.rs`: 4 weeks of weekly use → half-life 2.99 M s
(inside cap), final strength above the floor; the idle twin dies on
schedule; the cap is exact; strength untouchable by marks.

### The DeathLedger

A brain that loses memories silently is a black hole. Tombstones carry
their reason inline (`forget` / `consolidated` — a consolidated fact is
not lost, it lives inside the semantic summary); rows retain physically
prunes are ledged `{text, reason, at}` — `decay` (retained prune),
`sem-sync prune` (the SEM2 channel following the canonical store) — in a
persisted ring of 256, read back on hydrate (old files hydrate empty;
serde-tolerant everywhere). Surfaced as `stats.recentDeaths` and the
`death_reason` field on tombstone read-outs.

## M3 — compaction keeps its memory (`4351a00bf`)

`/compact` folds old turns into a summary in the context window — and the
summary used to exist **only there**. Now both drivers (embedded chat.js,
TUI chat.rs) land the distilled gist in the brain via
`sofuu.agent.rememberGist(target, text)`: shared scope as a first-class
record with provenance `compact`; agent scope under the agent's own
namespace; SEM2 mirror with the exact stored text. Compaction never
blocks on the brain. Measured: `tests/compact_gist_e2e.sh` — a two-turn
pty chat + `/compact` leaves exactly one `source:compact` record with the
summary's text.

## What was deliberately deferred

| Item | Why |
|---|---|
| M1.3 rare-token re-rank | Needs a token-level harness; the vector-only fixture cannot drive it. The measured M1 bar is already met. |
| Desktop memory command surface | Host-UI feature (the /brain info + /memory pages read `recentDeaths`/`death_reason` from the surfaces above once built). |
| KV-page store in the generation loop | Inference-workstream scale; the dormant single reference lives in quant.rs. |
| Reason for deaths across a pre-M2 tombstone file | Old tombstones hydrate with no death_reason — displayed as "unknown (written before M2)" on the read-out surfaces. |
