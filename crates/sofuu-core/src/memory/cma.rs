// sofuu-core — CMA (Cognitive Memory Architecture).
//
// Rust port of src/memory/mod_memory.c — the 4-tier memory model:
// working → episodic → semantic → entity, with Ebbinghaus decay,
// k-means consolidation, TF-IDF "dream" summarization, co-recall gravity,
// and resonance scoring. Pure logic (no QuickJS) — the JS bindings stay in
// C (mod_memory.c) via FFI, or can be re-exposed later.

use crate::memory::hnsw::Hnsw;
use std::collections::{HashMap, HashSet};

pub const TIER_WORKING: u8 = 0;
pub const TIER_EPISODIC: u8 = 1;
pub const TIER_SEMANTIC: u8 = 2;
pub const TIER_FORGOTTEN: u8 = 3;
pub const TIER_ENTITY: u8 = 4;

pub const GRAVITY_G: f32 = 0.05;
pub const CORECALL_MAX: usize = 8;
pub const DUP_DISTANCE: f32 = 0.05;

/// M1.3 (Brain Core II): the query-token re-rank ceiling. A single hit
/// carrying every rare query token moves up at most this much of its own
/// score — enough to flip top-1 over a near neighbor, never enough to
/// vault a clearly-wrong record into relevance.
///
/// W4.6 (QTSQ): the multiplicative re-rank this ceiling bounded was
/// replaced by reciprocal rank fusion (see `recall_tokens`), whose scores
/// are an order of magnitude smaller, so the ceiling holds a fortiori. The
/// constant is kept as the documented scale bound score-thresholding
/// callers may test against (the recall fixtures still assert it).
pub const TOKEN_RERANK_BOOST: f32 = 0.35;

/// W4.6 (QTSQ): the reciprocal-rank-fusion constant. Each channel
/// contributes `1 / (RRF_K + rank)` with 1-based ranks, so a first-place
/// vote is worth 1/61 and a tenth-place vote 1/70 — a consensus of fair
/// rankings beats one channel's runaway score, which is the whole point of
/// fusing BM25 + vector distance + recency. 60 is the value the plan
/// specifies (and the C module's constant).
pub const RRF_K: f32 = 60.0;
/// W4.6: BM25 saturation and length-normalisation (Robertson/Sparck Jones
/// defaults). Not tunable at runtime: the fusion must be reproducible.
pub const BM25_K1: f32 = 1.2;
/// W4.6: BM25 length-normalisation strength.
pub const BM25_B: f32 = 0.75;
/// W4.6: the reflex partition epsilon of the token path — a hit this close
/// to the query vector IS the query, and no lexical evidence may outrank it.
pub const TOKEN_REFLEX_EPS: f32 = 0.02;

/// B5.1 (ROADMAP-BRAIN-CMA): diversity floor. While filling top-k, a
/// candidate whose cosine to an already-picked hit is >= this is skipped
/// as a near-duplicate. The default is measured, not guessed -- see
/// `memory::recall_eval`. (The roadmap called this "MMR"; the harness
/// measured the textbook linear trade-off promoting unrelated records on
/// this scoring scale, so the shipped mechanism is MMR-family diversity by
/// suppression: relevance order is never inverted, near-duplicates are.)
pub const DIVERSITY_FLOOR_DEFAULT: f32 = 0.6;
/// B5.2: recency boost ceiling -- a just-written memory scores up to this
/// much above an equally-strong stale one, decaying with RECENCY_TAU.
pub const RECENCY_MAX_BOOST: f32 = 0.15;
/// B5.2: recency time constant (seven days).
pub const RECENCY_TAU_SECONDS: f32 = 7.0 * 86400.0;

/// B6 (ROADMAP-BRAIN-CMA): bit 30 flags a WORKING-SET id. Working records
/// live beside the persistent table (never in `records`, never in the HNSW
/// index, never serialized), so their ids must never collide with a
/// persistent record id — the flag makes a working id distinguishable
/// everywhere an id travels (recall hits, mark_positive, promotion). Bit
/// 30 rather than 31 so the id stays POSITIVE through every width it
/// crosses (i32 JS values, JSON numbers): 0x80000000 reads as negative in
/// JS and a JSON round-trip of it would drop the flag on the floor.
/// Persistent ids can never reach 2^30 (the store cap is 100k).
pub const WORKING_ID_FLAG: u32 = 0x4000_0000;
/// B6: session working-set cap. A runaway session cannot pin RAM: over
/// this, the OLDEST working records are dropped (they were never recalled
/// often enough to matter — promotion is what earns persistence).
pub const WORKING_MAX_RECORDS: usize = 256;

/// B6: a session-scoped working record. Held beside the persistent table:
/// not in `records`, not in the HNSW index, never written to disk. It is
/// recallable immediately (merged into recall with the 0.5 working tier
/// weight) and disappears on reload unless promoted.
#[derive(Clone, Debug)]
pub struct WorkingRecord {
    /// Stable session-local id (`WORKING_ID_FLAG | seq`) — stable because
    /// the cap evicts by created_at, which would renumber positions.
    pub id: u32,
    pub vector: Vec<f32>,
    pub text: String,
    pub role: String,
    pub kv_page_id: u32,
    /// B1.2 provenance carried through promotion (the persistent record
    /// keeps the ORIGINAL write time, not the promotion time).
    pub created_at: u32,
    pub source: String,
}

/// B6: `true` when `id` refers to the working set, not the persistent table.
pub fn is_working_id(id: u32) -> bool {
    id & WORKING_ID_FLAG != 0
}

/// M3 (PLAN-MEMORY-TOKENS): records below this strength are already
/// invisible to recall (the same floor filters in recall()); retain()
/// physically drops them so RAM and brain files stop accumulating
/// tombstones. Single source of truth for both sites.
pub const FORGET_FLOOR: f32 = 0.05;
/// Pinned facts (written by /remember) survive decay + retain().
pub const PIN_ROLE: &str = "user_pin";

/// W5.5 (QTSQ): the HONEST consolidation label. `consolidate()` folds a
/// cluster of weak episodic rows into one semantic survivor whose text is
/// an EXTRACTIVE pick — `dream_extract` returns a sentence that actually
/// existed in one of the members (current code; the C original synthesised
/// "Consolidated Semantic Fact (N episodes)", and this crate's pre-W5.5
/// pass stamped the survivor `role = "system"` with no way to tell it
/// apart from any other system-written row). The survivor is a
/// representative of real content, not a synthetic summary, so it is
/// labelled `representative` — a role that asserts exactly what it is and
/// nothing more.
pub const REPRESENTATIVE_ROLE: &str = "representative";

/// W5.5: `true` for a consolidation survivor (the row that stands for a
/// folded cluster). Ranking and decay treat it as ordinary semantic
/// content — this predicate exists so readers can SAY it is a
/// representative, not so they can score it differently.
pub fn is_representative(rec: &Record) -> bool {
    rec.role == REPRESENTATIVE_ROLE
}

/// H-08 (AUDIT-2026-10-29): every marker a PIN actually carries. The Rust
/// API and its tests write `role = PIN_ROLE`; the JS `/remember` path
/// writes `role = "user", source = "pin"` (agent.js `cma.remember(v, t,
/// 'user', 0, 'pin')`) and, for entity/turn pins, `entity_type = "pin"`.
/// A pin is exempt from the recall floor under ALL of them — a user who
/// explicitly asked to remember something must never find it silently
/// unreachable because one surface spelled the marker differently.
pub fn is_pinned(rec: &Record) -> bool {
    rec.role == PIN_ROLE
        || rec.role == "pin"
        || rec.source == "pin"
        || rec.entity_type.as_deref() == Some("pin")
}

/// H-08: the ranking strength a PIN is scored with. Decay is left alone
/// for every record (the e-folding curve is the documented law); what the
/// pin exemption changes is that a pinned fact is *ranked as if fresh*, so
/// the strength term can never bury an exact match under unrelated rows.
/// `strength` is capped at 1.0 everywhere else, so this is the ceiling.
pub const PIN_RANK_STRENGTH: f32 = 1.0;

/* ── W5.4: the pin floor is an INVARIANT, not just a ranking hint ─────
 *
 * H-08 floored a pin's *ranking* term (`PIN_RANK_STRENGTH`) and exempted
 * it from recall's floor and retain()'s prune — but the stored strength
 * still followed the ordinary curve all the way down (≈1e-9 after a
 * month). Anything that reads the stored number instead of the ranking
 * term therefore still saw a pin as "about to die": a brain card, an
 * eviction pass, a UI "weakest facts" list, a reader porting the file.
 * A pin must not merely *rank* as fresh; it must not decay under the
 * floor at all. `PIN_FLOOR` is that floor: on every tick a live pinned
 * row is clamped up to it, and pinning a weak row raises it immediately.
 */
pub const PIN_FLOOR: f32 = 0.2;

/* ── W5.3: the decay POLICY ───────────────────────────────────────────
 *
 * One hardcoded law used to serve every brain: `strength *= exp(-dt /
 * half_life)`. W5.3 makes the policy explicit and persistable — the
 * store keeps the `(tag, seconds)` pair `decay_params()` returns and
 * re-applies it with `set_decay_params()`. Exactly three tags exist;
 * the setter rejects everything else:
 *
 *   DECAY_TAG_NONE         strength never moves (a tick only ages rows).
 *   DECAY_TAG_EXPONENTIAL  strength *= 0.5^(dt / seconds) — `seconds` is
 *                          a HALF-LIFE: a 100 s half-life leaves 0.5 after
 *                          100 s and 0.25 after 200 s. `seconds == 0.0` is
 *                          the legacy sentinel meaning "use each record's
 *                          own `half_life`", which is the pre-W5.3 curve
 *                          and the DEFAULT, so an old brain file ticks
 *                          exactly as it always did.
 *   DECAY_TAG_TTL          nothing decays; a row strictly older than
 *                          `seconds` (measured from `created_at`) leaves
 *                          the live set at tick time.
 *
 * Every tag still enforces the W5.4 pin floor and never ages a
 * tombstone.
 */
pub const DECAY_TAG_NONE: u8 = 0;
pub const DECAY_TAG_EXPONENTIAL: u8 = 1;
pub const DECAY_TAG_TTL: u8 = 2;

/// W5.3: the DeathLedger reason a TTL expiry records. It reuses the same
/// tombstone → `retain()` accounting as `forget()` ("forget") and
/// consolidation ("consolidated"); the reason is what tells the three
/// apart once the row is physically gone.
pub const TTL_DEATH_REASON: &str = "ttl";

/// W5.4: the pin invariant in one place — a LIVE pinned row never carries
/// less than `PIN_FLOOR`. Called on every tick (whatever the policy) and
/// by `pin()` so the raise is immediate. A tombstone is left alone: a
/// forgotten row is dead, and no decay rule should dress it back up.
fn clamp_to_pin_floor(rec: &mut Record) {
    if rec.tier != TIER_FORGOTTEN && is_pinned(rec) && rec.strength < PIN_FLOOR {
        rec.strength = PIN_FLOOR;
    }
}

/// W5.3 TTL: `true` when the row is STRICTLY older than `seconds` by its
/// `created_at`. `created_at == 0` means "unknown" (a brain file written
/// before the field existed) and never expires — an unknown age is not
/// evidence of death. A `created_at` in the future (clock skew) reads as
/// age 0 through `saturating_sub`, not as a huge age.
fn ttl_expired(rec: &Record, now: u32, seconds: f64) -> bool {
    if rec.created_at == 0 {
        return false;
    }
    (now.saturating_sub(rec.created_at) as f64) > seconds
}

/// H-07: the vector near-duplicate gate's search width. Identical strings
/// never reach it — `Cma::text_index` answers those in O(1) — so this is
/// the *near*-duplicate path only. The audit's "ef=128 duplicate gate"
/// figure was `recall()`'s frontier (pool × 4 = 128), not this search.
/// Kept tiny on purpose, and the whole gate can be switched off for bulk
/// ingestion (`Cma::set_dup_gate(false)` / SOFUU_BRAIN_DUP_GATE=0) — the
/// exact-text index still merges identical facts either way.
pub const DUP_GATE_EF: usize = 8;

/// W5.1: how many near candidates the dual-key gate inspects. It used to
/// ask for exactly one: distance alone decided, so the nearest neighbour
/// WAS the merge target. With the text key in play the nearest vector can
/// veto a merge (a different fact that happens to embed close by) while
/// the row that DOES carry the same words sits at the same or a hair
/// longer distance — asking for one candidate would let the veto silently
/// disable dedup. Four is still a trivial search at `DUP_GATE_EF`, and it
/// is bounded: a merge only ever considers candidates inside the 0.05
/// radius that pass the text key.
pub const DUP_GATE_K: usize = 4;

/* ── W5.1 (QTSQ): the DUAL-KEY near-duplicate gate ────────────────────
 *
 * The shipped gate merged on VECTOR DISTANCE ALONE: two different facts
 * that happened to embed near each other collapsed into one (the loser's
 * text was silently dropped from every later recall). QTSQ W5.1 requires
 * BOTH keys before a merge:
 *
 *   · DISTANCE  `sqrt(||a-b||²) < DUP_DISTANCE` (0.05) — same place in
 *     embedding space, and
 *   · TEXT      token Jaccard similarity `>= DUP_TEXT_JACCARD` (0.85) —
 *     the same words, essentially.
 *
 * Only the conjunction merges; either key alone inserts. That is the
 * AND gate, and it is applied by every write path in this file:
 * `remember`/`remember_with_source` (stage 2 — stage 1 is exact text, where
 * Jaccard is 1.0 by construction), `import_records`, and
 * `remember_working`.
 *
 * NO-TEXT ROWS (documented exception): the text key needs tokens on BOTH
 * sides. When either side has none — a vector-only row, or a remember
 * called with an empty text — there is no text evidence to compare, so
 * the pair falls back to the distance key alone (the pre-W5.1 behaviour).
 * Vector-only rows therefore keep deduping exactly as before; they are
 * the one case where "same place" is all the evidence that exists. This
 * is deliberate and tested (`dual_key_gate_vector_only_rows_*`).
 */
pub const DUP_TEXT_JACCARD: f32 = 0.85;

/// W5.1: the TEXT key — set-based token Jaccard similarity in [0, 1],
/// using the same tokenization as the recall path (lowercased alphanumeric
/// runs; `tokenize`). `0.0` when either side has no tokens: an absent text
/// proves nothing, and the caller (the gate) then falls back to the
/// distance key alone rather than silently merging or silently splitting.
pub fn token_jaccard(a: &str, b: &str) -> f32 {
    let sa: HashSet<String> = tokenize(a).into_iter().collect();
    let sb: HashSet<String> = tokenize(b).into_iter().collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.len() + sb.len() - inter;
    inter as f32 / union as f32
}

/// W5.1: the text key's verdict for one candidate pair. `None` = the pair
/// carries no text evidence (at least one side is empty/untokenizable), in
/// which case the gate uses the distance key alone.
fn text_key_verdict(incoming: &str, existing: &str) -> Option<bool> {
    if tokenize(incoming).is_empty() || tokenize(existing).is_empty() {
        return None;
    }
    Some(text_jaccard_or_containment(incoming, existing) >= DUP_TEXT_JACCARD)
}

/// Jaccard, plus the restatement case: a text that keeps every content token
/// of the other and merely adds words to it ("session-only fact" ->
/// "session-only fact, restated", Jaccard 0.667) is the same fact said
/// again. An embedding collision between unrelated sentences shares no
/// token set at all, so containment cannot resurrect the merges the text
/// key exists to stop — that is what the `dual_key_gate_*` tests pin.
fn text_jaccard_or_containment(incoming: &str, existing: &str) -> f32 {
    let j = token_jaccard(incoming, existing);
    if j >= DUP_TEXT_JACCARD {
        return j;
    }
    let (a, b) = (tokenize(incoming), tokenize(existing));
    let (sa, sb): (std::collections::HashSet<&String>, std::collections::HashSet<&String>) =
        (a.iter().collect(), b.iter().collect());
    let (small, big) = if sa.len() <= sb.len() { (&sa, &sb) } else { (&sb, &sa) };
    if small.len() >= 2 && small.iter().all(|t| big.contains(t)) {
        1.0
    } else {
        j
    }
}

/// W5.1: the gate's text-key decision, with the documented no-text
/// fallback folded in. `true` = this pair passes the text key (including
/// the vector-only fallback); `false` = the text key VETOES the merge even
/// though the vectors are close.
fn text_key_passes(incoming: &str, existing: &str) -> bool {
    text_key_verdict(incoming, existing).unwrap_or(true)
}

/// H-07: recall's minimum search frontier on a corpus bigger than
/// `HNSW_SMALL_GRAPH`. The old default was `pool * 4` (128 for a top-5
/// query), which on a real graph is wide enough that the early-exit bound
/// never fires before most of the corpus has been walked — the
/// `0.9 ms + 1.07 µs · n` curve the audit measured. 48 keeps every slot
/// the ranking pass needs (the search may return the whole `pool`) while
/// terminating by width instead of by exhaustion.
pub const RECALL_EF_MIN: usize = 48;

/// A single memory record.
#[derive(Clone, Debug)]
pub struct Record {
    pub id: u32,
    pub kv_page_id: u32,
    pub strength: f32,
    pub age_seconds: u32,
    pub half_life: u32,
    pub tier: u8,
    pub text: String,
    pub role: String,
    pub entity_name: Option<String>,
    pub entity_type: Option<String>,
    /// Co-recall partners (gravity field).
    pub corecall_ids: Vec<u32>,
    pub total_recalls: u16,
    pub positive_recalls: u16,
    /// B1.2 provenance (ROADMAP-BRAIN-CMA): unix seconds. 0 = unknown
    /// (legacy files written before the fields existed).
    pub created_at: u32,
    /// Last time recall() returned this record; 0 = never.
    pub last_recalled_at: u32,
    /// Where the memory came from ("chat", "agent:researcher", "pin", …).
    /// Empty = unknown.
    pub source: String,
    /// M2 (Brain Core II): why this row died, once it is a tombstone
    /// ("forget" — the user asked; "consolidated" — folded into a semantic
    /// summary). Physically pruned rows live in the DeathLedger instead.
    /// None = alive (and None never enters a brain card).
    pub death_reason: Option<String>,
}

/// B1.2: unix seconds, saturating at u32 (the same width the page index
/// uses). A clock before the epoch reads as 0 = unknown.
fn now_secs() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().min(u32::MAX as u64) as u32)
        .unwrap_or(0)
}

/// B5.3: the recall search width from the environment
/// (`SOFUU_BRAIN_EF_SEARCH`); unset / 0 keeps the built-in default. Shared
/// by every surface that opens a brain (JS shell, C ABI facade).
pub fn ef_search_from_env() -> Option<usize> {
    std::env::var("SOFUU_BRAIN_EF_SEARCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// B5.1: cosine similarity, clamped to [-1, 1]; a zero-norm vector reads
/// as unrelated (0.0) instead of dividing by zero.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / denom).clamp(-1.0, 1.0)
}

impl Record {
    fn new(id: u32, text: &str, role: &str, kv_page_id: u32) -> Self {
        Self {
            id,
            kv_page_id,
            strength: 1.0,
            age_seconds: 0,
            half_life: 86400, // 1 day
            tier: TIER_EPISODIC,
            text: text.to_string(),
            role: role.to_string(),
            entity_name: None,
            entity_type: None,
            corecall_ids: Vec::new(),
            total_recalls: 0,
            positive_recalls: 0,
            created_at: now_secs(),
            last_recalled_at: 0,
            source: String::new(),
            death_reason: None,
        }
    }
}

/// M2 (Brain Core II): the DeathLedger — a persisted ring of every record
/// physically pruned from this store, with WHY it died. A decayed fact the
/// user later hunts for ("where did that note about X go?") deserves an
/// answer, not silence: "decayed below the recall floor on 2026-10-16" is
/// the difference between a brain and a black hole. Tombstones keep theirs
/// inline (`Record.death_reason`); rows retain() physically drops — where
/// no record survives to carry the field — land here.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DeathEntry {
    pub text: String,
    pub reason: String,
    pub at: u64,
}

/// The ledger keeps the most recent 256 deaths (a brain's distant past is
/// exactly the part nobody asks about; the list is diagnostics, not an
/// archive).
pub const DEATH_LEDGER_CAP: usize = 256;

/// M2: positive reinforcement buys decay tolerance. Each use at least
/// DOUBLES an episodic row's half-life (never less than +2 days — the SM-2
/// insight that interval growth, not a fixed multiplier, is what keeps a
/// cadence alive): the first use buys 3 days (a weekly cadence dips only
/// to exp(-7/3) ≈ 0.096, safely above the 0.05 floor, where the untouched
/// 1-day curve dies at exp(-7) ≈ 0.0009), and repeated use converges at
/// the tier cap — never sturdier than the semantic curve.
pub const EPISODIC_HALF_LIFE_CAP: u32 = 86400 * 30;

/// Semantic-tier growth cap (M2): semantic rows growth toward quarterly
/// decay, not immortality — entities (1-year) stay the durable tier.
pub const SEMANTIC_HALF_LIFE_CAP: u32 = 86400 * 90;

fn half_life_cap_for(tier: u8) -> u32 {
    if tier == TIER_EPISODIC {
        EPISODIC_HALF_LIFE_CAP
    } else {
        SEMANTIC_HALF_LIFE_CAP
    }
}

/// B1.1 read-out filter (ROADMAP-BRAIN-CMA). Every field is optional;
/// `limit == 0` means "no limit". `offset` pages through the filtered set
/// in record order.
#[derive(Clone, Debug, Default)]
pub struct ListFilter {
    pub tier: Option<u8>,
    pub role: Option<String>,
    pub entity: Option<String>,
    /// Case-insensitive substring match over the record text (the chat
    /// browser's `/memory <query>`).
    pub text: Option<String>,
    pub min_strength: Option<f32>,
    pub offset: usize,
    pub limit: usize,
}

/// A recall result.
#[derive(Clone, Debug)]
pub struct RecallHit {
    pub id: u32,
    pub distance: f32,
    /// The composite relevance score on the vector scale (strength × tier
    /// × resonance × recency, plus the pin floor). `recall_tokens` keeps
    /// this field's meaning so score-thresholding callers keep working —
    /// the fusion ORDER does not ride on it (see `fusion`).
    pub score: f32,
    pub role: String,
    pub text: String,
    pub tier: u8,
    pub strength: f32,
    /// Set for TIER_ENTITY records — the entity namespace the record
    /// belongs to (e.g. "agent:researcher"). Lets JS-side consumers filter
    /// per-agent scopes out of one physical brain (PLAN-AGENTS A3).
    pub entity: Option<String>,
    /// W4.6 (QTSQ): the reciprocal-rank-fusion score that ORDERED the token
    /// path — `Σ 1/(RRF_K + rank)` over the BM25, vector-distance and
    /// recency channels (plus the exact-phrase vote), so max ≈ 3/61 ≈
    /// 0.049 for three channels. `None` on the plain vector `recall` path
    /// (no lexical signal, nothing to fuse) and when the token path had
    /// fewer than two candidates to rank.
    pub fusion: Option<f32>,
}

/// B2.2: what a card merge did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImportStats {
    pub added: usize,
    /// Near-duplicates of existing records — reinforced, not duplicated.
    pub merged: usize,
    /// Records the card carried that could not be used (no text, or a
    /// tombstone).
    pub skipped: usize,
}

/// W5.2 (QTSQ): what ONE remember call did. The old surface returned a bare
/// i32 id, so a caller could not tell a fresh insert from a merge — nor
/// which existing id absorbed the fact. That information matters whenever
/// a write is followed by a mutation of "the record": silently merging
/// means the mutation lands on a row the caller never saw.
///
/// The JS-facing shape the plan specifies is `{ id, merged, into? }`;
/// `to_json()` emits exactly that (`into` is omitted when it does not
/// apply), and `id` is always the id of the row that now holds the fact —
/// so a merge reports `id == into.unwrap()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RememberOutcome {
    /// The id of the record that now holds the fact — the fresh row on an
    /// insert, the absorbing (surviving) row on a merge.
    pub id: u32,
    /// `true` when an existing live row was reinforced instead of a new
    /// row being appended.
    pub merged: bool,
    /// On a merge, the id of the row that absorbed the write (the same
    /// value as `id`); `None` on a fresh insert.
    pub into: Option<u32>,
}

impl RememberOutcome {
    /// W5.2: `{"id":N,"merged":bool}` for an insert, plus `"into":N` for a
    /// merge — the exact object shape the JS bridge should hand back.
    pub fn to_json(&self) -> String {
        match self.into {
            Some(into) => format!(
                "{{\"id\":{},\"merged\":{},\"into\":{}}}",
                self.id, self.merged, into
            ),
            None => format!("{{\"id\":{},\"merged\":{}}}", self.id, self.merged),
        }
    }
}

pub struct Cma {
    pub vec_dim: usize,
    /// W5.1 scope switch: TRUE only while `promote_working_by_ids` runs.
    /// Promotion moves a fact the session already accepted, and the working
    /// set's near-dup rule is distance-only by design (B6.1), so the text
    /// key must not veto it — "a third wording" promoted onto a stored
    /// "a second wording of the same fact" at the SAME vector is one fact.
    /// Ordinary writes keep the text key.
    pub text_gate_disabled: bool,
    pub index: Hnsw,
    pub records: Vec<Record>,
    /// B3.2 (ROADMAP-BRAIN-CMA): ids whose persisted snapshot is stale —
    /// recall reinforcement, a tier change, a tombstone, an entity upsert.
    /// A segmented flush appends exactly these (plus records the store has
    /// never seen), so a flush costs O(changed) instead of O(total).
    dirty_ids: Vec<u32>,
    /// A structural change the incremental path cannot express: retain()
    /// renumbered ids, or a store-side eviction. The next flush rewrites
    /// the live records into fresh segments (compaction).
    pub layout_dirty: bool,
    /// B5.3: HNSW search width for recall. None = the built-in default
    /// (`(pool * 4).max(8)`); larger trades latency for recall.
    pub ef_search: Option<usize>,
    /// B5.1: near-duplicate suppression floor (None = off, pure relevance).
    pub diversity_floor: Option<f32>,
    /// B6: the session working set — beside the persistent table, never
    /// serialized, merged into recall, promoted on use.
    pub working: Vec<WorkingRecord>,
    /// B6: monotonically increasing working-id sequence (eviction must not
    /// renumber ids a caller still holds).
    working_next_seq: u32,
    /// M2: the DeathLedger — why records physically pruned from this store
    /// died. Persisted as a top-level "deaths" array in the metadata (old
    /// files without it hydrate empty), newest last, capped.
    pub deaths: Vec<DeathEntry>,
    /// H-07 (AUDIT-2026-10-29): exact-text index — `text -> id` of the
    /// newest record carrying that string. Identical facts are resolved in
    /// O(1) (a hash lookup plus, at most, ONE distance to that record)
    /// instead of a graph search per insert. A cache of `records`: every
    /// mutation path that writes `records[i].text` (remember, forget,
    /// hydrate, import, consolidation, retain's renumbering) keeps it in
    /// sync, and `rebuild_text_index()` re-derives it after anything that
    /// bypasses those paths.
    text_index: HashMap<String, u32>,
    /// H-07: whether the VECTOR near-duplicate gate runs on insert (exact
    /// text is always gated — this is the fuzzy 0.05-radius check).
    /// Default on: "the same fact said differently" is the case it exists
    /// for. Bulk ingestion can turn it off (SOFUU_BRAIN_DUP_GATE=0) and
    /// lose nothing but paraphrase folding.
    pub dup_gate: bool,
    /// W5.3: the decay policy tag in force — always one of the
    /// `DECAY_TAG_*` constants (the only writer is `set_decay_params`,
    /// which rejects every other value, so an unknown tag cannot exist
    /// inside a live store).
    decay_tag: u8,
    /// W5.3: the policy's seconds parameter — the half-life for
    /// DECAY_TAG_EXPONENTIAL (`0.0` = the per-record legacy curve) and the
    /// TTL for DECAY_TAG_TTL (`0.0` = unused, DECAY_TAG_NONE).
    decay_seconds: f64,
}

/// Above this many dirty ids, tracking per-record stops paying — the next
/// flush compacts instead (one pass over live records).
const DIRTY_TRACK_CAP: usize = 4096;

/// One record's JSON metadata — exactly the format the C brain files use
/// (`cma_flush` in mod_memory.c), so files stay interchangeable.
/// The brain-file metadata document: `{"records":[...], "deaths":[...]?}`.
/// Parsed once, directly — `records` is required (a file without it is not a
/// brain file), `deaths` is optional (files written before the ledger).
#[derive(serde::Deserialize)]
struct MetaDoc {
    records: Vec<RecordJson>,
    #[serde(default)]
    deaths: Vec<DeathEntry>,
}

/// One record's JSON metadata — exactly the format the C brain files use
/// (`cma_flush` in mod_memory.c), so files stay interchangeable.
#[derive(serde::Serialize, serde::Deserialize)]
struct RecordJson {
    vector_index: u32,
    kv_page_id: u32,
    strength: f32,
    // The old C cma_flush didn't persist age_seconds/half_life — default
    // them so legacy brain files still hydrate.
    #[serde(default)]
    age_seconds: u32,
    #[serde(default = "default_half_life")]
    half_life: u32,
    tier: u8,
    text: String,
    role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entity_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entity_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    corecall_ids: Vec<u32>,
    // P3 (AUDIT-2026-09-07): corecall_count/corecall_rpos were dropped —
    // both were derivable from corecall_ids (len, len % 8) and never read
    // back on hydrate. serde ignores unknown fields, so brain files written
    // with them still parse.
    total_recalls: u16,
    positive_recalls: u16,
    /* B1.2 provenance — optional so every brain written before this
     * existed still hydrates (0 / "" = unknown). */
    #[serde(default)]
    created_at: u32,
    #[serde(default)]
    last_recalled_at: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    source: String,
    // M2 (Brain Core II) — why a tombstone died. Optional everywhere: old
    // files lack it, alive records never carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    death_reason: Option<String>,
}

fn default_half_life() -> u32 {
    86400
}

impl Cma {
    pub fn new(vec_dim: usize) -> Self {
        Self {
            vec_dim,
            text_gate_disabled: false,
            index: Hnsw::new(vec_dim),
            records: Vec::new(),
            dirty_ids: Vec::new(),
            layout_dirty: false,
            ef_search: None,
            diversity_floor: Some(DIVERSITY_FLOOR_DEFAULT),
            working: Vec::new(),
            working_next_seq: 0,
            deaths: Vec::new(),
            text_index: HashMap::new(),
            /* H-07: SOFUU_BRAIN_DUP_GATE=0 / false / off disables the fuzzy
             * gate for a bulk load; unset keeps the shipped default. An
             * explicit `set_dup_gate` always wins afterwards. */
            dup_gate: std::env::var("SOFUU_BRAIN_DUP_GATE")
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    !(v == "0" || v == "false" || v == "off" || v == "no")
                })
                .unwrap_or(true),
            /* W5.3: the shipped default is the historical law — the
             * per-record e-folding curve. An old brain file that predates
             * the policy field hydrates into exactly this, and a new brain
             * ticks the same way until `set_decay_params` says otherwise. */
            decay_tag: DECAY_TAG_EXPONENTIAL,
            decay_seconds: 0.0,
        }
    }

    /// H-07: turn the vector near-duplicate gate on/off. Exact-text
    /// duplicates always merge — this only controls the 0.05-radius fuzzy
    /// check that costs one graph search per insert.
    pub fn set_dup_gate(&mut self, on: bool) {
        self.dup_gate = on;
    }

    /// W5.3: the decay policy this brain ticks with, as the `(tag, seconds)`
    /// pair the store persists. A fresh brain answers
    /// `(DECAY_TAG_EXPONENTIAL, 0.0)` — the historical per-record curve
    /// (see `DECAY_TAG_EXPONENTIAL` for the `0.0` sentinel).
    pub fn decay_params(&self) -> (u8, f64) {
        (self.decay_tag, self.decay_seconds)
    }

    /// W5.3: install a decay policy. Returns `false` — and leaves the
    /// current policy untouched — for an unknown tag, for a non-finite or
    /// negative `seconds`, and for a TTL that is not strictly positive.
    /// `DECAY_TAG_NONE` ignores `seconds` (normalised to `0.0`).
    ///
    /// `DECAY_TAG_EXPONENTIAL` accepts `seconds == 0.0`: that is the
    /// documented "use each record's own `half_life`" sentinel, i.e. the
    /// default, so persisting `decay_params()` and re-applying it after a
    /// hydrate is a lossless round trip in every mode (including the
    /// default one).
    pub fn set_decay_params(&mut self, tag: u8, seconds: f64) -> bool {
        if !seconds.is_finite() || seconds < 0.0 {
            return false;
        }
        match tag {
            DECAY_TAG_NONE => {
                self.decay_tag = DECAY_TAG_NONE;
                self.decay_seconds = 0.0;
                true
            }
            DECAY_TAG_EXPONENTIAL => {
                /* 0.0 = the per-record curve; > 0.0 = a global half-life. */
                self.decay_tag = DECAY_TAG_EXPONENTIAL;
                self.decay_seconds = seconds;
                true
            }
            DECAY_TAG_TTL => {
                if seconds <= 0.0 {
                    return false;
                }
                self.decay_tag = DECAY_TAG_TTL;
                self.decay_seconds = seconds;
                true
            }
            _ => false,
        }
    }

    /// H-07: re-derive the exact-text index from `records`. Cheap (O(n) hash
    /// inserts) and only needed after a path that rewrote text outside
    /// `remember`/`forget`/`import_records` — retain's renumbering, hydrate
    /// and consolidation call it themselves.
    pub fn rebuild_text_index(&mut self) {
        self.text_index.clear();
        for rec in &self.records {
            if rec.text.is_empty() {
                continue;
            }
            /* Alive rows win over tombstones so a re-said fact reinforces
             * the live one; among equals the LATER id (newest sighting)
             * wins, matching what remember() would have stored. */
            let prev = self.text_index.get(&rec.text).copied();
            if let Some(prev) = prev {
                if rec.tier == TIER_FORGOTTEN
                    && self
                        .records
                        .get(prev as usize)
                        .map(|p| p.tier != TIER_FORGOTTEN)
                        .unwrap_or(false)
                {
                    continue;
                }
            }
            self.text_index.insert(rec.text.clone(), rec.id);
        }
    }

    /// H-07: the record an exact-text insert would resolve to, if any.
    fn live_id_for_text(&self, text: &str) -> Option<u32> {
        if text.is_empty() {
            return None;
        }
        let id = *self.text_index.get(text)?;
        match self.records.get(id as usize) {
            Some(rec) if rec.tier != TIER_FORGOTTEN => Some(id),
            _ => None,
        }
    }

    /// B5.3: set the HNSW recall search width (None = built-in default).
    pub fn set_ef_search(&mut self, ef: Option<usize>) {
        self.ef_search = ef.filter(|n| *n > 0);
    }

    /// B5.1: set the near-duplicate suppression floor (None = off).
    pub fn set_diversity_floor(&mut self, floor: Option<f32>) {
        self.diversity_floor = floor.map(|f| f.clamp(0.0, 1.0));
    }

    /// B6.1: write a session working record. Recallable immediately (recall
    /// merges the working set with the 0.5 tier weight), never serialized —
    /// a reload shows no working records. A near-duplicate WITHIN the
    /// working set reinforces the existing copy (same fact said twice in a
    /// session is one fact); a near-duplicate in the PERSISTENT store is
    /// deliberately not gated here — the twin is suppressed at recall by
    /// the diversity floor, and promotion merges it on use.
    ///
    /// W5.1: "near-duplicate" means the DUAL KEY here too — close vector
    /// AND (token Jaccard >= `DUP_TEXT_JACCARD` | either side textless).
    /// Two different session scratch notes that merely embed nearby now
    /// stay two records.
    pub fn remember_working(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
        source: &str,
    ) -> u32 {
        debug_assert_eq!(vec.len(), self.vec_dim);
        /* Working-set near-dup gate (same DUP_DISTANCE + W5.1 text key as
         * the persistent one): the newer sighting refreshes the timestamp
         * so the fact stays fresh, and the caller gets the SAME id back.
         *
         * ONE exception, and it is the scratchpad's whole point: an
         * IDENTICAL vector (d == 0) is the same fact however it is
         * re-worded. The persistent store's 0.85 Jaccard guard exists to
         * stop two different sentences that embed close together from
         * collapsing; it has nothing to say when the embeddings are
         * bit-identical, and applying it there broke B6.1 ("the same fact
         * within the session is ONE working record") for the ordinary case
         * of restating a sentence — 'session-only fact' vs 'session-only
         * fact, restated' scores Jaccard 0.667 and used to get a NEW id. */
        for w in &mut self.working {
            let d = sq_dist(w.vector.as_slice(), vec).sqrt();
            if (d == 0.0 || d < DUP_DISTANCE && text_key_passes(text, &w.text)) {
                w.created_at = now_secs();
                if !source.is_empty() {
                    w.source = source.to_string();
                }
                return w.id;
            }
        }
        let id = WORKING_ID_FLAG | self.working_next_seq;
        self.working_next_seq = self.working_next_seq.wrapping_add(1);
        self.working.push(WorkingRecord {
            id,
            vector: vec.to_vec(),
            text: text.to_string(),
            role: role.to_string(),
            kv_page_id,
            created_at: now_secs(),
            source: source.to_string(),
        });
        /* Cap: drop the OLDEST (never-promoted scratch, by definition —
         * promotion removes records as it promotes them). */
        if self.working.len() > WORKING_MAX_RECORDS {
            let oldest = self
                .working
                .iter()
                .enumerate()
                .min_by_key(|(_, w)| (w.created_at, w.id))
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.working.remove(oldest);
        }
        id
    }

    /// B6: how many session working records are live.
    pub fn working_len(&self) -> usize {
        self.working.len()
    }

    /// B6: drop the whole working set (session end). Returns how many
    /// records were dropped. Unpromoted facts die here — that is the
    /// design: nothing persists unless a turn actually used it.
    pub fn clear_working(&mut self) -> usize {
        let n = self.working.len();
        self.working.clear();
        n
    }

    /// B6.2: promote the working records with these ids into the persistent
    /// store as episodic — the turn-commit hook for "working records the
    /// turn actually used". The ids are the recall-hit ids of this turn
    /// (channel-split by the caller, exactly like markPositive: canonical
    /// ids go to the canonical store, sem-only ids to the sem store).
    /// Persistent ids in the list are ignored (already persisted). A
    /// near-duplicate of an existing persistent record reinforces it
    /// (merged) instead of duplicating; a promoted record keeps its
    /// ORIGINAL created_at and source. Returns (promoted, merged); the
    /// working copies are removed either way.
    pub fn promote_working_by_ids(&mut self, ids: &[u32]) -> (usize, usize) {
        if ids.is_empty() || self.working.is_empty() {
            return (0, 0);
        }
        let mut promoted = 0usize;
        let mut merged = 0usize;
        /* Collect then promote: remember_with_source mutates `records`,
         * not `working`, so borrowing the vectors first is enough. */
        let matches: Vec<WorkingRecord> = self
            .working
            .iter()
            .filter(|w| ids.contains(&w.id))
            .cloned()
            .collect();
        self.text_gate_disabled = true;
        for w in matches {
            let before = self.records.len();
            let new_id = self.remember_with_source(
                &w.vector,
                &w.text,
                &w.role,
                w.kv_page_id,
                &w.source,
            );
            if new_id < 0 {
                continue;
            }
            let id = new_id as u32;
            if self.records.len() > before && (id as usize) == before {
                /* A fresh record — remember_with_source stamped now, so
                 * restore the working record's ORIGINAL write time:
                 * provenance (and the recency factor) reflect when the
                 * fact was actually said, not when it graduated. */
                self.records[id as usize].created_at = w.created_at;
                self.mark_dirty(id);
                promoted += 1;
            } else {
                /* The dup gate reinforced an existing persistent record. */
                merged += 1;
            }
            self.working.retain(|w2| w2.id != w.id);
        }
        self.text_gate_disabled = false;
        (promoted, merged)
    }

    /// Mark one record's persisted snapshot stale (B3.2).
    fn mark_dirty(&mut self, id: u32) {
        if self.layout_dirty {
            return; /* a compaction will cover it anyway */
        }
        if self.dirty_ids.len() >= DIRTY_TRACK_CAP {
            self.layout_dirty = true;
            self.dirty_ids.clear();
            return;
        }
        if !self.dirty_ids.contains(&id) {
            self.dirty_ids.push(id);
        }
    }

    /// Ids changed since the last flush, ascending.
    pub fn take_dirty(&mut self) -> Vec<u32> {
        let mut out = std::mem::take(&mut self.dirty_ids);
        out.sort_unstable();
        out
    }

    /// How many records carry an unpersisted change (diagnostics).
    pub fn dirty_count(&self) -> usize {
        self.dirty_ids.len()
    }

    /// The store has persisted everything up to (and including) the current
    /// records — clear the dirty set.
    pub fn mark_persisted(&mut self) {
        self.dirty_ids.clear();
        self.layout_dirty = false;
    }

    /// Hydrate from a loaded brain file: the flat f32 vector table + the
    /// metadata JSON (C's `cma_qtsq_load` output). Returns None on a
    /// malformed payload (caller falls back to a fresh store).
    pub fn hydrate(&mut self, vecs: &[f32], n: usize, dim: usize, meta_json: &str) -> bool {
        self.hydrate_with_snapshot(vecs, n, dim, meta_json, None)
    }

    /// W3.1: `hydrate`, but with the chance to ADOPT a persisted graph
    /// snapshot instead of rebuilding the index by inserting every vector.
    ///
    /// The rebuild is what made a 20k-record open take 21 s and a 100k one
    /// 170 s: `add_vector` runs a full HNSW search per record. A snapshot of
    /// the same generation carries the topology, so hydrating from it is
    /// O(records) memcpy rather than O(records × log n) distance work.
    ///
    /// The snapshot is only used when it PROVES it belongs to this data —
    /// same record count, same dimension — otherwise the rebuild happens as
    /// before. A stale or corrupt snapshot must degrade to the old path, never
    /// to a wrong answer.
    pub fn hydrate_with_snapshot(
        &mut self,
        vecs: &[f32],
        n: usize,
        dim: usize,
        meta_json: &str,
        snapshot: Option<crate::memory::hnsw::Hnsw>,
    ) -> bool {
        if dim != self.vec_dim || vecs.len() != n * dim {
            return false;
        }
        // The brain-file metadata is {"records":[...]} — the same shape the
        // C cma_flush wrote (M2: plus an optional top-level "deaths").
        let prof = std::env::var_os("SOFUU_PROFILE").is_some();
        let mut t = std::time::Instant::now();
        /* One parse, straight into the typed document.
         *
         * This used to build a serde_json::Value and then CLONE the records
         * array out of it and parse that again — two full passes and a deep
         * copy of every record's metadata (measured: 51 ms at 20k records,
         * i.e. ~255 ms of a 100k open, paid for nothing). A missing
         * "records" key still fails the hydrate, and unknown keys are still
         * ignored, so the accepted format is unchanged. */
        let doc: MetaDoc = match serde_json::from_str(meta_json) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let t_parse = t.elapsed().as_secs_f64() * 1000.0;
        t = std::time::Instant::now();
        let records: Vec<RecordJson> = doc.records;
        let deaths: Vec<DeathEntry> = doc.deaths;
        /* W3.1: with a valid snapshot the index comes from the file; only the
         * record table needs building. */
        let adopt = snapshot.filter(|snap| snap.len() == n);
        let rebuild_index = adopt.is_none();
        for i in 0..n {
            if rebuild_index {
                self.index.add_vector(&vecs[i * dim..(i + 1) * dim]);
            }
            let mut rec = Record::new(i as u32, "", "", 0);
            if let Some(rj) = records.get(i) {
                rec.kv_page_id = rj.kv_page_id;
                rec.strength = rj.strength;
                rec.age_seconds = rj.age_seconds;
                // ml-3 (AUDIT-2026-09-07): a hostile file can carry
                // half_life 0 — every decay tick would become exp(-inf) = 0,
                // an instant permanent erase. Normalize to the 1-day default.
                rec.half_life = if rj.half_life == 0 {
                    default_half_life()
                } else {
                    rj.half_life
                };
                rec.tier = rj.tier;
                rec.text = rj.text.clone();
                rec.role = rj.role.clone();
                rec.entity_name = rj.entity_name.clone();
                rec.entity_type = rj.entity_type.clone();
                rec.corecall_ids = rj.corecall_ids.clone();
                rec.total_recalls = rj.total_recalls;
                rec.positive_recalls = rj.positive_recalls;
                rec.created_at = rj.created_at;
                rec.last_recalled_at = rj.last_recalled_at;
                rec.source = rj.source.clone();
                rec.death_reason = rj.death_reason.clone();
            } else {
                rec.text = "(corrupt record)".to_string();
            }
            self.records.push(rec);
        }
        /* M2: the DeathLedger rides beside the records (old files without
         * it hydrate an empty ledger). Ring-capped on READ too, so a
         * hand-edited or hostile file cannot balloon it. */
        if !deaths.is_empty() {
            self.deaths = deaths;
            if self.deaths.len() > DEATH_LEDGER_CAP {
                self.deaths = self.deaths.split_off(self.deaths.len() - DEATH_LEDGER_CAP);
            }
        }
        let t_build = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        /* H-07: the exact-text index is derived state — re-derive it from
         * whatever the file carried (legacy files included). */
        self.rebuild_text_index();
        if prof {
            eprintln!(
                "[prof] hydrate: parse {t_parse:.1} build {t_build:.1} text_index {:.1} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
        if let Some(snap) = adopt {
            self.index = snap;
        }
        true
    }

    /// Serialize records to the brain-file metadata JSON (matches C).
    pub fn records_json(&self) -> String {
        self.records_value().to_string()
    }

    /// P3 (AUDIT-2026-09-07): the same document as a Value — callers that
    /// post-process the metadata take it directly instead of the old
    /// serialize-to-String-then-reparse round trip.
    pub fn records_value(&self) -> serde_json::Value {
        let arr: Vec<RecordJson> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| RecordJson {
                vector_index: i as u32,
                kv_page_id: r.kv_page_id,
                strength: r.strength,
                age_seconds: r.age_seconds,
                half_life: r.half_life,
                tier: r.tier,
                text: r.text.clone(),
                role: r.role.clone(),
                entity_name: r.entity_name.clone(),
                entity_type: r.entity_type.clone(),
                corecall_ids: r.corecall_ids.clone(),
                total_recalls: r.total_recalls,
                positive_recalls: r.positive_recalls,
                created_at: r.created_at,
                last_recalled_at: r.last_recalled_at,
                source: r.source.clone(),
                death_reason: r.death_reason.clone(),
            })
            .collect();
        let mut doc = serde_json::json!({ "records": arr });
        if !self.deaths.is_empty() {
            doc["deaths"] = serde_json::to_value(&self.deaths).unwrap_or(serde_json::Value::Null);
        }
        doc
    }

    /// KV hints: the distinct kv_page_ids of the top-n nearest memories.
    /// Reinforces strength on use (matches C's cma_kv_hints).
    pub fn kv_hints(&mut self, query: &[f32], n: usize) -> Vec<u32> {
        if self.records.is_empty() {
            return Vec::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        let ef = (n * 3).max(16);
        let hits = self.index.search(query, ef);
        for (id, _) in hits {
            if self.records[id as usize].tier == TIER_FORGOTTEN {
                continue;
            }
            let kv = self.records[id as usize].kv_page_id;
            if kv == 0 {
                continue;
            }
            if seen.insert(kv) {
                /* The use-refresh must mark the snapshot stale too — a
                 * flush-less mutation would persist stale strengths when
                 * nothing else dirtied the record this turn. */
                {
                    let rec = &mut self.records[id as usize];
                    rec.strength = 1.0;
                    /* M2: growth under the tier cap, like mark_positive —
                     * the old ×1.2 here was unbounded (a hot KV row could
                     * reach a thousand-day half-life). */
                    if rec.half_life > 0 {
                        let cap = half_life_cap_for(rec.tier);
                        if rec.half_life < cap {
                            let grown = ((rec.half_life as f32) * 1.2) as u32;
                            rec.half_life = rec.half_life.max(grown.min(cap));
                        }
                    }
                }
                self.mark_dirty(id);
                out.push(kv);
                if out.len() >= n {
                    break;
                }
            }
        }
        out
    }

    /// B1.1 read-out: the filtered, paginated record view every surface
    /// (chat browser, brain server, C ABI) shares. Returns a JSON object
    /// `{total, offset, limit, records:[…]}` — `total` is the FILTERED
    /// count before pagination, so a caller can render "N of M".
    pub fn list_json(&self, filter: &ListFilter) -> String {
        let filtered: Vec<&Record> = self
            .records
            .iter()
            /* Tombstones are hidden unless explicitly asked for — a
             * browser listing "forgotten" rows by default is noise, and
             * they are pruned at the next turn boundary anyway. */
            .filter(|r| match filter.tier {
                Some(t) => r.tier == t,
                None => r.tier != TIER_FORGOTTEN,
            })
            .filter(|r| filter.role.as_deref().map_or(true, |role| r.role == role))
            .filter(|r| {
                filter
                    .entity
                    .as_deref()
                    .map_or(true, |e| r.entity_name.as_deref() == Some(e))
            })
            .filter(|r| {
                filter.text.as_deref().map_or(true, |needle| {
                    !needle.is_empty()
                        && r.text.to_lowercase().contains(&needle.to_lowercase())
                })
            })
            .filter(|r| filter.min_strength.map_or(true, |m| r.strength >= m))
            .collect();
        let total = filtered.len();
        let page: Vec<serde_json::Value> = filtered
            .into_iter()
            .skip(filter.offset)
            .take(if filter.limit == 0 { usize::MAX } else { filter.limit })
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "tier": r.tier,
                    "strength": r.strength,
                    "role": r.role,
                    "text": r.text,
                    "entity": r.entity_name,
                    "entity_type": r.entity_type,
                    "created_at": r.created_at,
                    "last_recalled_at": r.last_recalled_at,
                    "source": r.source,
                    "age_seconds": r.age_seconds,
                    "total_recalls": r.total_recalls,
                    "positive_recalls": r.positive_recalls,
                    "kv_page_id": r.kv_page_id,
                    /* M2: present on tombstones (why they died); absent on
                     * living rows. */
                    "death_reason": r.death_reason,
                })
            })
            .collect();
        serde_json::json!({
            "total": total,
            "offset": filter.offset,
            "limit": filter.limit,
            "records": page,
        })
        .to_string()
    }

    /// B2.2: merge a card's records + vectors into this store. The caller
    /// has ALREADY re-embedded the texts when the card's space differs (the
    /// embedder lives at the binding layer), so `vecs` here are always in
    /// THIS store's space. Merge policy: a near-duplicate reinforces the
    /// existing record (max strength, earliest created_at wins, original
    /// text/role/source kept); everything else inserts with the card's tier,
    /// role, entity and provenance. Tombstones are never imported.
    ///
    /// `records_json` is a JSON OBJECT `{"records":[…]}` — one record per
    /// vector, in vector order (W6.3; a bare array does not match here and
    /// yields the all-zero stats, which is why the JS cfunc now rejects it).
    pub fn import_records(
        &mut self,
        vecs: &[f32],
        n: usize,
        dim: usize,
        records_json: &str,
    ) -> ImportStats {
        let mut stats = ImportStats::default();
        if dim != self.vec_dim || vecs.len() < n * dim {
            return stats;
        }
        let Ok(root) = serde_json::from_str::<serde_json::Value>(records_json) else {
            return stats;
        };
        let Some(arr) = root.get("records").and_then(|r| r.as_array()) else {
            return stats;
        };
        for (i, rj) in arr.iter().enumerate().take(n) {
            let v = &vecs[i * dim..(i + 1) * dim];
            let Some(text) = rj.get("text").and_then(|t| t.as_str()) else {
                stats.skipped += 1;
                continue;
            };
            if text.trim().is_empty() {
                stats.skipped += 1;
                continue;
            }
            let their_tier = rj
                .get("tier")
                .and_then(|t| t.as_u64())
                .unwrap_or(TIER_EPISODIC as u64) as u8;
            if their_tier == TIER_FORGOTTEN {
                stats.skipped += 1;
                continue;
            }
            /* A card cannot import session state or an unknown tier:
             * working (0) is a process-local construct by design, and a
             * hostile tier > 4 would render as garbage in every reader.
             * Both land episodic. */
            let their_tier = if their_tier == TIER_WORKING || their_tier > TIER_ENTITY {
                TIER_EPISODIC
            } else {
                their_tier
            };
            let their_strength = rj
                .get("strength")
                .and_then(|s| s.as_f64())
                .unwrap_or(1.0) as f32;
            /* A FUTURE created_at is hostile (saturating_sub in the
             * recency factor would grant it the maximum freshness boost
             * forever) — treat it as unknown, the neutral value. */
            let mut their_created = rj
                .get("created_at")
                .and_then(|c| c.as_u64())
                .unwrap_or(0) as u32;
            if their_created > now_secs() {
                their_created = 0;
            }

            /* Dedup: a near-duplicate of an existing live record merges.
             * H-07: same two-stage policy as remember() — the exact-text
             * index first (O(1), where the text key is 1.0 by
             * construction), then the bounded vector gate. W5.1: the
             * vector gate is DUAL-KEY here too — a close vector alone must
             * not swallow a card's different fact. */
            if let Some(dup_id) = self.live_id_for_text(text) {
                if sq_dist(self.index.vector(dup_id), v).sqrt() < DUP_DISTANCE {
                    let rec = &mut self.records[dup_id as usize];
                    if their_strength > rec.strength {
                        rec.strength = their_strength;
                    }
                    if their_created > 0 && (rec.created_at == 0 || their_created < rec.created_at) {
                        rec.created_at = their_created; /* earliest sighting wins */
                    }
                    stats.merged += 1;
                    continue;
                }
            }
            let dup = if self.records.is_empty() || !self.dup_gate {
                None
            } else {
                self.index
                    .search_with_ef(v, DUP_GATE_K, DUP_GATE_EF)
                    .into_iter()
                    .find(|(id, d)| {
                        let rec = &self.records[*id as usize];
                        *d < DUP_DISTANCE
                            && rec.tier != TIER_FORGOTTEN
                            && (self.text_gate_disabled || text_key_passes(text, &rec.text))
                    })
            };
            if let Some((dup_id, _)) = dup {
                let rec = &mut self.records[dup_id as usize];
                if their_strength > rec.strength {
                    rec.strength = their_strength;
                }
                if their_created > 0 && (rec.created_at == 0 || their_created < rec.created_at) {
                    rec.created_at = their_created; /* earliest sighting wins */
                }
                stats.merged += 1;
                continue;
            }

            let id = self.records.len() as u32;
            let role = rj.get("role").and_then(|r| r.as_str()).unwrap_or("imported");
            let mut rec = Record::new(id, text, role, 0);
            rec.tier = their_tier;
            rec.strength = their_strength.clamp(0.0, 1.0);
            rec.created_at = their_created;
            rec.source = rj
                .get("source")
                .and_then(|s| s.as_str())
                .unwrap_or("import")
                .to_string();
            rec.entity_name = rj.get("entity").and_then(|e| e.as_str()).map(str::to_string);
            rec.entity_type = rj
                .get("entity_type")
                .and_then(|e| e.as_str())
                .map(str::to_string);
            if rec.tier == TIER_ENTITY {
                rec.half_life = 86400 * 365;
            }
            self.records.push(rec);
            self.index.add_vector(v);
            /* H-07: keep the exact-text index complete — an imported fact is
             * a real record, and a later remember() of the same words must
             * resolve it in O(1) instead of a graph search. */
            if !text.is_empty() {
                self.text_index.insert(text.to_string(), id);
            }
            stats.added += 1;
        }
        stats
    }

    /// The row-major flat f32 table for EVERY record (n × vec_dim) — the
    /// brain-file shape, where the table stays aligned with `records_value`
    /// (tombstones included: the file is the store, not a view).
    pub fn flat_vectors(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.records.len() * self.vec_dim);
        for i in 0..self.records.len() {
            out.extend_from_slice(self.index.vector(i as u32));
        }
        out
    }

    /// B2: the flat table for the records a VIEW selects, in view order —
    /// an export card must pair `list_json(filter).records[i]` with
    /// `vectors[i*dim..]`. Using the unfiltered table here silently
    /// misaligned every card that had tombstones in the store.
    pub fn flat_vectors_for(&self, filter: &ListFilter) -> Vec<f32> {
        let mut out = Vec::new();
        for r in self.records.iter().filter(|r| match filter.tier {
            Some(t) => r.tier == t,
            None => r.tier != TIER_FORGOTTEN,
        }) {
            out.extend_from_slice(self.index.vector(r.id));
        }
        out
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remember a fact. Near-duplicate detection reinforces instead.
    ///
    /// Thin i32 wrapper kept for every existing caller (the C ABI facade,
    /// the fuzz harness, the decay/recall fixtures): it returns the same id
    /// `remember_outcome` reports. New callers that need to know whether
    /// the write merged — and which id absorbed it — should call
    /// `remember_outcome` (W5.2).
    pub fn remember(&mut self, vec: &[f32], text: &str, role: &str, kv_page_id: u32) -> i32 {
        self.remember_outcome(vec, text, role, kv_page_id).id as i32
    }

    /// W5.2: remember, reporting insert-vs-merge. Same policy as
    /// `remember_with_source`, no provenance.
    pub fn remember_outcome(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
    ) -> RememberOutcome {
        self.remember_with_source_outcome(vec, text, role, kv_page_id, "")
    }

    /// B1.2: remember + record where the memory came from. A near-duplicate
    /// keeps its ORIGINAL source (the first sighting is the provenance).
    ///
    /// Thin i32 wrapper — see `remember` for the rich variant.
    pub fn remember_with_source(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
        source: &str,
    ) -> i32 {
        self.remember_with_source_outcome(vec, text, role, kv_page_id, source)
            .id as i32
    }

    /// W5.2: the rich remember — reports whether the write inserted or
    /// merged, and which row absorbed it.
    ///
    /// H-07 (AUDIT-2026-10-29): the duplicate decision is made in two
    /// stages, cheapest first —
    ///   1. EXACT TEXT: `text_index` resolves the same string to its record
    ///      in O(1); one distance to that record settles it (`DIM` flops,
    ///      no search). Identical facts therefore never touch the graph.
    ///      The text key is satisfied by construction here (Jaccard 1.0);
    ///      only the distance key can still veto.
    ///   2. DUAL-KEY GATE (W5.1): only when the text is new (or the same
    ///      text embeds somewhere else) does the 0.05-radius HNSW probe
    ///      run, at `DUP_GATE_EF` (k = `DUP_GATE_K`) and only if `dup_gate`
    ///      is on —
    ///      and the candidate now has to pass BOTH `dist < DUP_DISTANCE`
    ///      AND `token_jaccard >= DUP_TEXT_JACCARD`. The pre-W5.1 path
    ///      merged on distance alone, so two unrelated memories that
    ///      shared a neighbourhood collapsed into one.
    /// Before this, EVERY insert ran stage 2 across the whole index — the
    /// measured 409–410 facts/s.
    pub fn remember_with_source_outcome(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
        source: &str,
    ) -> RememberOutcome {
        debug_assert_eq!(vec.len(), self.vec_dim);
        /* Stage 1: exact text. */
        if let Some(dup_id) = self.live_id_for_text(text) {
            let same_place = sq_dist(self.index.vector(dup_id), vec).sqrt() < DUP_DISTANCE;
            if same_place {
                self.reinforce_duplicate(dup_id);
                return RememberOutcome {
                    id: dup_id,
                    merged: true,
                    into: Some(dup_id),
                };
            }
            /* Same words, different embedding (re-said in a new context):
             * not a duplicate by construction — fall through so the vector
             * gate can still fold it into a near neighbour, and let the
             * insert record the newer sighting in the text index. */
        }
        /* Stage 2: dual-key near-duplicate gate (optional, bounded). The
         * nearest FEW neighbours are inspected because a closer row can
         * VETO the merge on the text key while the row that really is the
         * same fact sits just behind it. */
        if self.dup_gate && !self.records.is_empty() {
            let hits = self.index.search_with_ef(vec, DUP_GATE_K, DUP_GATE_EF);
            let target = hits.iter().copied().find(|(dup_id, dist)| {
                let rec = &self.records[*dup_id as usize];
                /* W5.1: the AND gate. A close vector alone is NOT a
                 * duplicate; the text has to agree too (or be absent on
                 * one side — the documented vector-only fallback). */
                *dist < DUP_DISTANCE
                    && rec.tier != TIER_FORGOTTEN
                    && (self.text_gate_disabled || text_key_passes(text, &rec.text))
            });
            if let Some((dup_id, _)) = target {
                self.reinforce_duplicate(dup_id);
                return RememberOutcome {
                    id: dup_id,
                    merged: true,
                    into: Some(dup_id),
                };
            }
        }
        let id = self.records.len() as u32;
        let mut rec = Record::new(id, text, role, kv_page_id);
        rec.source = source.to_string();
        self.records.push(rec);
        self.index.add_vector(vec);
        if !text.is_empty() {
            self.text_index.insert(text.to_string(), id);
        }
        self.mark_dirty(id);
        RememberOutcome {
            id,
            merged: false,
            into: None,
        }
    }

    /// The merge arm shared by both gate stages: the surviving row is
    /// refreshed (a re-said fact is a fresh fact) and its snapshot marked
    /// stale. The survivor keeps its own text, role and provenance.
    fn reinforce_duplicate(&mut self, dup_id: u32) {
        let rec = &mut self.records[dup_id as usize];
        rec.strength = 1.0;
        rec.age_seconds = 0;
        self.mark_dirty(dup_id);
    }

    /// Recall top-k memories with composite scoring (strength × tier ×
    /// resonance), gravity boost, and reinforcement.
    ///
    /// H-07 (AUDIT-2026-10-29): the candidate set comes from the HNSW
    /// graph search at a BOUNDED width. The old default frontier was
    /// `pool * 4` (128 for a top-5 query), which is wide enough that on a
    /// real graph the early-exit bound almost never fires — the search
    /// walked most of the corpus and the latency curve read as a linear
    /// scan (`0.9 ms + 1.07 µs · n`). Two documented fallbacks remain:
    ///   · a corpus at or below `HNSW_SMALL_GRAPH` is searched EXACTLY
    ///     (frontier = n): small brains get the true top-k, and every
    ///     ranking fixture keeps its pinned numbers;
    ///   · an explicit `set_ef_search` / SOFUU_BRAIN_EF_SEARCH always
    ///     wins, so an operator can still buy recall with latency.
    pub fn recall(&mut self, query: &[f32], top_k: usize) -> Vec<RecallHit> {
        if self.records.is_empty() && self.working.is_empty() {
            return Vec::new();
        }
        /* Candidate pool: the search returns the closest `pool` records. */
        let pool = (top_k * 3).max(32);
        let n = self.index.len();
        let ef = match self.ef_search {
            Some(ef) => ef,
            None if n <= crate::memory::hnsw::HNSW_SMALL_GRAPH => n.max(1),
            None => pool.max(RECALL_EF_MIN),
        };
        let mut hits = self.index.search_with_ef(query, pool, ef);
        let now = now_secs();
        // Filter forgotten + too-weak, score.
        let mut scored: Vec<(u32, f32, f32)> = Vec::new(); // (id, dist, score)
        for (id, dist) in hits.drain(..) {
            let rec = &self.records[id as usize];
            /* H-08: a PIN is never filtered by strength. It decays like
             * everything else (the curve is unchanged), but an explicitly
             * remembered fact must stay recallable forever — the audit
             * found pins dropped from recall after ~3 idle days while
             * retain() kept them on disk: retained, but unreachable. */
            let pinned = is_pinned(rec);
            if rec.tier == TIER_FORGOTTEN || (!pinned && rec.strength < FORGET_FLOOR) {
                continue;
            }
            let tier_w = match rec.tier {
                TIER_WORKING => 0.5,
                TIER_SEMANTIC => 1.0,
                TIER_ENTITY => 1.5,
                _ => 0.8,
            };
            let norm = dist / (dist + 1.0);
            /* H-08: rank a pin as if it had never decayed. Lifecycle
             * (decay, retention, on-disk strength) is untouched — only the
             * ordering term is floored, so an aged pin cannot be buried by
             * a fresh, unrelated row with a tiny cosine edge. */
            let rank_strength = if pinned {
                rec.strength.max(PIN_RANK_STRENGTH)
            } else {
                rec.strength
            };
            let mut score = (1.0 - norm) * rank_strength * tier_w;
            // Resonance: engagement history boost (max ~30%).
            if rec.total_recalls > 0 {
                let ratio = ((rec.positive_recalls as f32 + 1.0).ln())
                    / ((rec.total_recalls as f32 + 2.0).ln());
                score *= 1.0 + ratio * 0.3;
            }
            /* B5.2: recency — a fresh memory outranks an equally-strong
             * stale one. created_at == 0 (legacy / unknown) is neutral. */
            if rec.created_at > 0 {
                let age = now.saturating_sub(rec.created_at) as f32;
                score *= 1.0 + RECENCY_MAX_BOOST * (-age / RECENCY_TAU_SECONDS).exp();
            }
            scored.push((id, dist, score));
        }
        /* B6.1: merge the session working set. It is small (capped at
         * WORKING_MAX_RECORDS), so a linear scan beats indexing it; the
         * scoring is the same composite (strength 1.0, the 0.5 working
         * tier weight, the recency boost — no resonance history). */
        for w in &self.working {
            let dist = sq_dist(w.vector.as_slice(), query).sqrt();
            let norm = dist / (dist + 1.0);
            let mut score = (1.0 - norm) * 1.0 * 0.5; /* TIER_WORKING weight */
            let age = now.saturating_sub(w.created_at) as f32;
            score *= 1.0 + RECENCY_MAX_BOOST * (-age / RECENCY_TAU_SECONDS).exp();
            scored.push((w.id, dist, score));
        }
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Gravity: co-recall partners of the top-1 get a boost. Working
        // records have no corecall field — only persistent ids take part.
        if scored.len() > 1 && !is_working_id(scored[0].0) {
            let top1 = scored[0].0;
            let partners: Vec<u32> = self.records[top1 as usize].corecall_ids.clone();
            for (id, _, score) in scored.iter_mut().skip(1) {
                if partners.contains(id) {
                    *score *= 1.0 + GRAVITY_G;
                }
            }
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        }

        /* M0 (Brain Core II): REFLEXIVITY — a query vector that (nearly)
         * equals a stored record's vector IS that record's question, so it
         * outranks every inexact hit no matter how its strength decayed.
         * Without this, fresh high-tier neighbors (entities weigh 1.5) bury
         * an exact match once decay pulls its strength under ~0.4: the fuzz
         * showed a 5-day-old fact missing from its OWN top-8 while unrelated
         * entities filled it. Ties inside a bucket stay score-ordered. */
        const REFLEX_EPS: f32 = 0.02;
        scored.sort_by(|a, b| {
            let ea = a.1 <= REFLEX_EPS;
            let eb = b.1 <= REFLEX_EPS;
            eb.cmp(&ea)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        /* B5.1: diversity — the top-k should not be five paraphrases of
         * one fact. Fills from the relevance-ordered pool, skipping
         * near-duplicates of records already picked. */
        if let Some(floor) = self.diversity_floor {
            if scored.len() > 1 {
                scored = self.diversify(scored, top_k, floor);
            }
        }

        let n = scored.len().min(top_k);
        let mut out = Vec::with_capacity(n);
        let mut recalled_ids: Vec<u32> = Vec::with_capacity(n);
        for (id, dist, score) in scored.into_iter().take(n) {
            if is_working_id(id) {
                /* B6.1: a working hit — no reinforcement (the record dies
                 * with the session unless promoted), no gravity, no dirty
                 * tracking. The id stays flagged so every consumer can
                 * tell which tier served it. */
                if let Some(w) = self.working.iter().find(|w| w.id == id) {
                    out.push(RecallHit {
                        id,
                        distance: dist,
                        score,
                        role: w.role.clone(),
                        text: w.text.clone(),
                        tier: TIER_WORKING,
                        strength: 1.0,
                        entity: None,
                        fusion: None,
                    });
                }
                continue;
            }
            // Reinforce + track recall.
            let rec = &mut self.records[id as usize];
            rec.strength = (rec.strength * 1.1).min(1.0);
            rec.last_recalled_at = now;
            if rec.total_recalls < 65535 {
                rec.total_recalls += 1;
            }
            recalled_ids.push(id);
            out.push(RecallHit {
                id,
                distance: dist,
                score,
                role: rec.role.clone(),
                text: rec.text.clone(),
                tier: rec.tier,
                strength: rec.strength,
                entity: rec.entity_name.clone(),
                fusion: None,
            });
            /* After the record borrow ends: its persisted snapshot is now
             * stale (strength + recall counters moved). */
            self.mark_dirty(id);
        }
        /* Gravity only ever sees persistent ids (working ids are filtered
         * above, so update_gravity cannot index out of bounds). */
        self.update_gravity(&recalled_ids);
        out
    }

    /// M1.3 (Brain Core II) + W4.6 (QTSQ): the token-aware recall — the
    /// recall for callers that also HAVE the query text. Vector similarity
    /// is the retrieval engine, but a chatty generic record can outrank the
    /// one record carrying the query's distinctive token ("quokka"): the
    /// embedder smooths over exactly the tokens that make a hit THE answer.
    ///
    /// W4.6 replaced M1.3's multiplicative rarity bonus with a real HYBRID
    /// FUSION: the candidate pool is ranked by RECIPROCAL RANK FUSION over
    /// three channels, with RRF_K = 60 (see the constants above),
    ///
    ///   rrf(h) = Σ_ch 1 / (RRF_K + rank_ch(h))
    ///     BM25    lexical relevance of h's text to the query tokens,
    ///             computed over the candidate pool (N = pool size,
    ///             k1 = BM25_K1, b = BM25_B,
    ///             idf = ln(1 + (N − df + 0.5)/(df + 0.5)))
    ///     VECTOR  L2 distance to the query (nearest first)
    ///     RECENCY created_at (newest first; 0 = unknown ranks last)
    ///
    /// Ranks are 1-based and ties SHARE the best rank of their group, so a
    /// channel with no evidence (all scores equal) hands every candidate
    /// rank 1 and cannot reorder anything. Two guards ride on the fused
    /// order — policy, documented, deterministic:
    ///
    ///   · EXACT-PHRASE BOOST — a hit whose text contains the query's
    ///     whole token phrase contiguously (>= 2 tokens) gets one extra
    ///     first-place vote, `1/(RRF_K + 1)`, and sorts above the plain
    ///     fused order (below the reflex partition).
    ///   · LEXICAL DOMINANCE — when exactly one hit uniquely carries a
    ///     query token that appears in no other candidate (df == 1) and
    ///     has strictly the greatest idf-weighted query coverage, it is
    ///     the answer the embedder smoothed over: it sorts above the
    ///     fused order (below the reflex partition). With no df == 1
    ///     token, or a coverage tie, the rule is inert.
    ///
    /// The M1.3 invariants survive: the reflex partition (distance <=
    /// TOKEN_REFLEX_EPS) still outranks everything, and a query with no
    /// lexical evidence is EXACTLY inert (every channel ties, ranks are
    /// equal, the vector order is preserved by construction) — both are
    /// pinned by tests, not just asserted here.
    ///
    /// `RecallHit::score` keeps its PRE-W4.6 meaning exactly: composite
    /// relevance × the M1.3 rarity multiplier (capped at
    /// +TOKEN_RERANK_BOOST). That matters because score is a threshold
    /// input, not just a rank key (agent.js gates recall at
    /// `score >= 0.30`); RRF values (≤ 3/61 ≈ 0.05) would silently pass
    /// nothing through such a gate. The fusion value that decides the
    /// ORDER rides in `RecallHit::fusion` — the two are deliberately
    /// different scales, and score is NOT rank-monotone on this path
    /// (sort by `fusion` to reconstruct the order, never by `score`).
    pub fn recall_tokens(
        &mut self,
        query: &[f32],
        top_k: usize,
        query_text: &str,
    ) -> Vec<RecallHit> {
        /* ≥3 chars: two-letter words ride idf≈0 anyway, but dropping them
         * keeps the rarity/BM25 work off the noise floor. Duplicates are
         * folded — a query term counts once in BM25 and once in coverage. */
        let mut q_tokens: Vec<String> = Vec::new();
        for t in tokenize(query_text) {
            if t.chars().count() >= 3 && !q_tokens.contains(&t) {
                q_tokens.push(t);
            }
        }
        if q_tokens.is_empty() {
            return self.recall(query, top_k);
        }
        let candidate_k = if top_k < 5 { (top_k * 3).max(6) } else { top_k };
        let mut hits = self.recall(query, candidate_k);
        if hits.len() < 2 {
            hits.truncate(top_k);
            return hits;
        }
        let order = self.fuse_hybrid(&hits, &q_tokens, top_k);
        let mut out = Vec::with_capacity(order.len());
        for (idx, fused, score) in order {
            let mut h = hits[idx].clone();
            h.fusion = Some(fused);
            h.score = score;
            out.push(h);
        }
        out
    }

    /// W4.6: the fusion pass behind `recall_tokens`. Returns (hit index,
    /// fused score, M1.3-scale score) in final order, at most `top_k` rows.
    /// See `recall_tokens` for the law; this function is pure ranking — it
    /// mutates nothing.
    fn fuse_hybrid(
        &self,
        hits: &[RecallHit],
        q_tokens: &[String],
        top_k: usize,
    ) -> Vec<(usize, f32, f32)> {
        let n = hits.len();

        /* ── channel 1: vector distance (nearest first) ─────────────── */
        let dist: Vec<f32> = hits.iter().map(|h| h.distance).collect();
        let rank_vector = competition_ranks(&dist, false);

        /* ── channel 2: BM25 over the candidate pool ────────────────── */
        let docs: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
        let bm25 = bm25_scores(&docs, q_tokens);
        let rank_bm25 = competition_ranks(&bm25, true);

        /* ── channel 3: recency (newest first; 0 = unknown, ranks last) ─ */
        let created: Vec<u32> = hits.iter().map(|h| self.created_at_of(h.id)).collect();
        let rank_recency = competition_ranks(&created, true);

        /* The lexical-dominance guard: M1.3's rarity statistic, now used
         * only to detect a UNIQUE carrier of a distinctive query token
         * (df == 1) with strictly the greatest idf-weighted coverage. */
        let toks: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
        let mut df: HashMap<&str, usize> = HashMap::new();
        for t in q_tokens {
            df.insert(t.as_str(), toks.iter().filter(|d| d.iter().any(|x| x == t)).count());
        }
        let coverage: Vec<f32> = toks
            .iter()
            .map(|doc| {
                let mut uniq: Vec<&String> = Vec::new();
                for x in doc {
                    if q_tokens.iter().any(|q| q == x) && !uniq.contains(&x) {
                        uniq.push(x);
                    }
                }
                uniq.iter()
                    .map(|t| 1.0 - (*df.get(t.as_str()).unwrap_or(&0) as f32 / n as f32))
                    .sum()
            })
            .collect();
        let max_cov = coverage.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let at_max: Vec<usize> = (0..n).filter(|&i| coverage[i] >= max_cov - 1e-6).collect();
        let dominance = if at_max.len() == 1 {
            let i = at_max[0];
            let carries_rare = toks[i]
                .iter()
                .any(|t| q_tokens.contains(t) && df.get(t.as_str()).copied().unwrap_or(0) == 1);
            if carries_rare && max_cov > 0.0 {
                Some(i)
            } else {
                None
            }
        } else {
            None
        };

        struct Row {
            idx: usize,
            fused: f32,
            priority: u8,
            rank_vector: usize,
            rank_bm25: usize,
            /// W4.6: the M1.3-boosted composite score, kept so consumers
            /// that threshold on `score` (agent.js gates at 0.30) see the
            /// same scale they always did. It is a RELEVANCE value, not a
            /// rank key — the order comes from `fused`.
            score: f32,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(n);
        for i in 0..n {
            let phrase = phrase_match(&hits[i].text, q_tokens);
            /* The pre-W4.6 score: composite relevance × the M1.3 rarity
             * multiplier, capped at +TOKEN_RERANK_BOOST. */
            let mult = (1.0 + TOKEN_RERANK_BOOST * coverage[i]).min(1.0 + TOKEN_RERANK_BOOST);
            let score = hits[i].score * mult;
            let mut fused = 1.0 / (RRF_K + rank_vector[i] as f32)
                + 1.0 / (RRF_K + rank_bm25[i] as f32)
                + 1.0 / (RRF_K + rank_recency[i] as f32);
            if phrase {
                /* The exact-phrase boost: one extra FIRST-PLACE vote. */
                fused += 1.0 / (RRF_K + 1.0);
            }
            /* Partition priority: reflex exact > phrase > lexical
             * dominance > plain fused order. The RRF score orders WITHIN
             * a band; the guards are policy on top of the arithmetic. */
            let priority = if hits[i].distance <= TOKEN_REFLEX_EPS {
                0
            } else if phrase {
                1
            } else if dominance == Some(i) {
                2
            } else {
                3
            };
            rows.push(Row {
                idx: i,
                fused,
                priority,
                rank_vector: rank_vector[i],
                rank_bm25: rank_bm25[i],
                score,
            });
        }
        /* Deterministic total order: priority, then fused score, then the
         * channel ranks (vector before lexical — retrieval is still the
         * engine), then id. */
        rows.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| {
                    b.fused
                        .partial_cmp(&a.fused)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.rank_vector.cmp(&b.rank_vector))
                .then_with(|| a.rank_bm25.cmp(&b.rank_bm25))
                .then_with(|| hits[a.idx].id.cmp(&hits[b.idx].id))
        });
        rows.into_iter()
            .take(top_k)
            .map(|r| (r.idx, r.fused, r.score))
            .collect()
    }

    /// W4.6: the write time behind a recall id (0 = unknown), for the
    /// recency channel of the fusion. Working records carry their own.
    fn created_at_of(&self, id: u32) -> u32 {
        if is_working_id(id) {
            return self
                .working
                .iter()
                .find(|w| w.id == id)
                .map(|w| w.created_at)
                .unwrap_or(0);
        }
        self.records.get(id as usize).map(|r| r.created_at).unwrap_or(0)
    }

    /// B6: the vector behind a recall id — the HNSW store for persistent
    /// records, the working record's own copy for working ids (the
    /// diversity re-rank compares across both tiers).
    fn vector_of(&self, id: u32) -> &[f32] {
        if is_working_id(id) {
            /* An evicted working id reads as unrelated (empty slice). */
            return self
                .working
                .iter()
                .find(|w| w.id == id)
                .map(|w| w.vector.as_slice())
                .unwrap_or(&[]);
        }
        self.index.vector(id)
    }

    /// B5.1: fill the top-k from the relevance-ordered pool, skipping any
    /// candidate that is a near-duplicate (cosine >= floor) of a record
    /// already picked. Relevance order is never inverted: a clearly better
    /// record is never displaced by a worse but dissimilar one. (The
    /// textbook linear MMR trade-off did exactly that on the evaluation
    /// fixture -- recall 1.00 -> 0.17 with unrelated records promoted; the
    /// numbers are in `memory::recall_eval`.)
    ///
    /// M1 (Brain Core II): once the distinct directions run out, the
    /// remaining slots take the suppressed candidates in relevance order
    /// instead of going home early. Returning fewer than top_k was the B5.1
    /// cure for paraphrase crowding, but on the flipped side of the same
    /// coin it STARVES recall whenever distinct topics are scarce relative
    /// to k -- the evaluated trade is now recall full-width, with the
    /// diversity pass first in line so crowding only re-enters at the tail.
    fn diversify(
        &self,
        pool: Vec<(u32, f32, f32)>,
        top_k: usize,
        floor: f32,
    ) -> Vec<(u32, f32, f32)> {
        let mut out: Vec<(u32, f32, f32)> = Vec::with_capacity(top_k.min(pool.len()));
        let mut deferred: Vec<(u32, f32, f32)> = Vec::new();
        for cand in pool {
            if out.len() >= top_k {
                break;
            }
            let dup = out.iter().any(|(sid, _, _)| {
                cosine_sim(self.vector_of(cand.0), self.vector_of(*sid)) >= floor
            });
            if dup {
                deferred.push(cand);
            } else {
                out.push(cand);
            }
        }
        for cand in deferred {
            if out.len() >= top_k {
                break;
            }
            out.push(cand);
        }
        out
    }

    /// W5.3/W5.4: one decay tick under the policy installed by
    /// `set_decay_params`.
    ///
    /// The law per tag:
    ///   · `DECAY_TAG_NONE`        — strength does not move at all; the
    ///     tick only ages the row.
    ///   · `DECAY_TAG_EXPONENTIAL` — `strength *= 0.5^(dt / seconds)`,
    ///     a HALF-life law: at a 100 s half-life, 100 s leaves 0.5 and
    ///     200 s leaves 0.25 (two half-lives). NOTE the arithmetic: the
    ///     0.3679 (= 1/e) figure is the ONE E-FOLD point, which this law
    ///     reaches at ln2 · seconds ≈ 69.3 s of a 100 s half-life — it is
    ///     NOT the 200 s value. With `seconds == 0.0` (the default) the
    ///     per-record curve of every pre-W5.3 brain applies instead:
    ///     `strength *= exp(-dt / half_life)`.
    ///   · `DECAY_TAG_TTL`         — nothing decays; a row strictly older
    ///     than `seconds` by `created_at` leaves the live set at tick
    ///     time. It leaves it the way every other death leaves it — a
    ///     tombstone (`TIER_FORGOTTEN` + its why), which `retain()` then
    ///     physically prunes and ledgers. No second removal path, and no
    ///     renumbering mid-tick (ids stay valid until the caller's next
    ///     turn boundary).
    ///
    /// W5.4 is unconditional on every tick, whatever the policy: a live
    /// pinned row is clamped UP to `PIN_FLOOR` and is never expired or
    /// tombstoned by decay.
    pub fn decay_tick(&mut self, dt_seconds: u32) {
        let (tag, seconds) = (self.decay_tag, self.decay_seconds);
        /* TTL measures wall age, so read the clock once for the whole
         * pass: a record cannot expire because the loop straddled a
         * second boundary. */
        let now = now_secs();
        let mut expired: Vec<u32> = Vec::new();
        for idx in 0..self.records.len() {
            let rec = &mut self.records[idx];
            if rec.tier == TIER_FORGOTTEN {
                continue;
            }
            rec.age_seconds = rec.age_seconds.saturating_add(dt_seconds);
            match tag {
                DECAY_TAG_NONE => {}
                DECAY_TAG_TTL => {
                    if !is_pinned(rec) && ttl_expired(rec, now, seconds) {
                        /* Positional, like every id in this store. */
                        expired.push(idx as u32);
                    }
                }
                /* EXPONENTIAL — the only value `set_decay_params` stores
                 * besides the two above, so the wildcard is exhaustive by
                 * construction. */
                _ => {
                    if seconds > 0.0 {
                        rec.strength *= 0.5f32.powf(dt_seconds as f32 / seconds as f32);
                    } else if rec.half_life == 0 {
                        /* ml-3 (AUDIT-2026-09-07) belt: half_life 0 would
                         * make the factor exp(-inf) = 0 — an instant
                         * permanent erase. hydrate normalizes the file
                         * value; decay stays multiplicative-neutral, not
                         * fatal. */
                    } else {
                        rec.strength *= (-(dt_seconds as f32) / rec.half_life as f32).exp();
                    }
                }
            }
            /* W5.4: the pin floor. Under every policy, on every tick. */
            clamp_to_pin_floor(rec);
        }
        /* TTL expiries leave the live set through the SAME accounting
         * `forget`/`consolidate` use — tombstone + why — so the ring
         * (and any reader) can tell a TTL death from a user forget. */
        for id in expired {
            if let Some(rec) = self.records.get_mut(id as usize) {
                rec.tier = TIER_FORGOTTEN;
                rec.death_reason = Some(TTL_DEATH_REASON.to_string());
                let gone = rec.text.clone();
                if !gone.is_empty() && self.text_index.get(&gone) == Some(&id) {
                    self.text_index.remove(&gone);
                }
                self.mark_dirty(id);
            }
        }
    }

    /// Mark memories as positively reinforced (user continued). M2: the
    /// counters move (strength untouched — I7), and the row's half-life
    /// doubles (≥ +2 days) — recognition buys decay tolerance: a memory
    /// the user keeps re-engaging survives its own absences instead of
    /// dying between uses. Growth caps at the tier's curve (episodic
    /// 30 d, semantic 90 d — entities need no help).
    pub fn mark_positive(&mut self, ids: &[u32]) -> u32 {
        let mut marked = 0;
        for &id in ids {
            if let Some(rec) = self.records.get_mut(id as usize) {
                if rec.positive_recalls < 65535 {
                    rec.positive_recalls += 1;
                    marked += 1;
                }
                if rec.half_life > 0 {
                    let cap = half_life_cap_for(rec.tier);
                    if rec.half_life < cap {
                        let doubled = rec.half_life.saturating_mul(2);
                        let stepped = rec.half_life.saturating_add(86400 * 2);
                        let grown = doubled.max(stepped).min(cap);
                        rec.half_life = rec.half_life.max(grown);
                        self.mark_dirty(id);
                    }
                }
            }
            if (id as usize) < self.records.len() {
                self.mark_dirty(id);
            }
        }
        marked
    }

    pub fn forget(&mut self, id: u32) -> bool {
        if let Some(rec) = self.records.get_mut(id as usize) {
            rec.tier = TIER_FORGOTTEN;
            rec.death_reason = Some("forget".to_string());
            /* H-07: a tombstoned row must not resolve an exact-text insert —
             * forgetting a fact and remembering it again is a NEW sighting.
             * `live_id_for_text` checks the tier anyway; dropping the entry
             * here keeps the index from accumulating one per forgotten row. */
            let gone = rec.text.clone();
            if !gone.is_empty() && self.text_index.get(&gone) == Some(&id) {
                self.text_index.remove(&gone);
            }
            self.mark_dirty(id);
            true
        } else {
            false
        }
    }

    /// W5.4: pin an existing live row so decay can never demote it. The
    /// marker written is the same one the JS `/remember` path writes
    /// (`source = "pin"`), which `is_pinned` already recognises and which
    /// `RecordJson` already persists — no schema change, and the pin
    /// survives a flush/hydrate round trip. A row below `PIN_FLOOR` is
    /// raised to it IMMEDIATELY (not at the next tick). Returns `false`
    /// for an id that names no record or a tombstone.
    pub fn pin(&mut self, id: u32) -> bool {
        if id as usize >= self.records.len() || self.records[id as usize].tier == TIER_FORGOTTEN {
            return false;
        }
        {
            let rec = &mut self.records[id as usize];
            if rec.source != "pin" {
                rec.source = "pin".to_string();
            }
            clamp_to_pin_floor(rec);
        }
        self.mark_dirty(id);
        true
    }

    /// W5.4: release a pin — every spelling `is_pinned` accepts is cleared
    /// (role `user_pin`/`pin` → `user`, `source = "pin"` → empty, a
    /// `entity_type = "pin"` marker → `None`), so the row decays on the
    /// ordinary curve again from wherever the floor left it. Returns
    /// `false` when the id names no record or the row was not pinned.
    pub fn unpin(&mut self, id: u32) -> bool {
        if id as usize >= self.records.len() {
            return false;
        }
        {
            let rec = &mut self.records[id as usize];
            if !is_pinned(rec) {
                return false;
            }
            if rec.role == PIN_ROLE || rec.role == "pin" {
                rec.role = "user".to_string();
            }
            if rec.source == "pin" {
                rec.source.clear();
            }
            if rec.entity_type.as_deref() == Some("pin") {
                rec.entity_type = None;
            }
        }
        self.mark_dirty(id);
        true
    }


    /// M2: record a physical death in the ledger (retain paths; the record
    /// itself is about to vanish, so the ring is the only witness). Newest
    /// last, ring-capped.
    fn ledger_death(&mut self, text: &str, reason: &str) {
        if self.deaths.len() >= DEATH_LEDGER_CAP {
            self.deaths.remove(0);
        }
        self.deaths.push(DeathEntry {
            text: text.chars().take(200).collect(),
            reason: reason.to_string(),
            at: now_secs() as u64,
        });
        self.layout_dirty = true;
    }

    /// M3 (PLAN-MEMORY-TOKENS): physically drop dead records so RAM and
    /// brain files actually shrink — TIER_FORGOTTEN tombstones, plus
    /// non-pinned records whose strength decayed below the recall floor.
    /// Entities and pinned facts (role "user_pin") survive unconditionally.
    ///
    /// Contract: call ONLY at a turn boundary. Survivors renumber, so any
    /// outstanding record ids (recall hits awaiting mark_positive) go stale;
    /// the JS caller runs this before recall each turn, never mid-turn.
    pub fn retain(&mut self) -> usize {
        let before = self.records.len();
        if before == 0 {
            return 0;
        }
        let is_dead = |rec: &Record| match rec.tier {
            TIER_FORGOTTEN => true,
            TIER_ENTITY => false,
            /* H-08: `is_pinned`, not just the Rust-side PIN_ROLE — the JS
             * /remember path marks pins with `source = "pin"`, and a pin
             * that retain() pruned was silently unrecoverable (the audit's
             * "retained on disk, invisible to recall" was really "kept by
             * role, dropped by any other spelling"). */
            _ => rec.strength < FORGET_FLOOR && !is_pinned(rec),
        };
        /* Nothing dead → keep records/vectors/index exactly as they are
         * (the cheap no-op path a healthy brain hits every turn). */
        if !self.records.iter().any(is_dead) {
            return 0;
        }
        let mut kept_records: Vec<Record> = Vec::with_capacity(before);
        let mut kept_vectors: Vec<Vec<f32>> = Vec::with_capacity(before);
        let mut remap: HashMap<u32, u32> = HashMap::new();
        /* Deaths are collected during the drain and ledged after — the
         * ledger borrows self mutably, and the drain already does. */
        let mut dead_notes: Vec<(String, String)> = Vec::new();
        for rec in self.records.drain(..) {
            if is_dead(&rec) {
                /* M2: a tombstone's own reason survives it ("forget" /
                 * "consolidated"); a weak non-pin pruned by strength is a
                 * decay death. */
                let reason = rec
                    .death_reason
                    .clone()
                    .unwrap_or_else(|| "decay".to_string());
                /* R-26: a consolidated row was ALREADY ledged by
                 * consolidate() when it became a tombstone — the physical
                 * prune must not enter it twice. Every other death is only
                 * observable here (the row is about to vanish). */
                if rec.death_reason.as_deref() != Some("consolidated") {
                    dead_notes.push((rec.text.clone(), reason));
                }
                continue;
            }
            remap.insert(rec.id, kept_records.len() as u32);
            kept_vectors.push(self.index.vector(rec.id).to_vec());
            kept_records.push(rec);
        }
        for (text, reason) in &dead_notes {
            self.ledger_death(text, reason);
        }
        // Renumber + remap co-recall partners onto the new id space.
        for (new_id, rec) in kept_records.iter_mut().enumerate() {
            rec.id = new_id as u32;
            rec.corecall_ids = rec
                .corecall_ids
                .iter()
                .filter_map(|id| remap.get(id).copied())
                .collect();
        }
        self.records = kept_records;
        // Rebuild the HNSW store over survivors (ids are positions again).
        self.index = Hnsw::new(self.vec_dim);
        for v in &kept_vectors {
            self.index.add_vector(v);
        }
        /* H-07: every id moved — the exact-text index is positional too. */
        self.rebuild_text_index();
        /* B3.2: survivors renumbered — every persisted id is stale, so the
         * incremental path cannot express this. The next flush compacts. */
        self.dirty_ids.clear();
        self.layout_dirty = true;
        before - self.records.len()
    }

    /// B3.5 (ROADMAP-BRAIN-CMA): keep only records whose text appears in
    /// `keep`, renumbering survivors exactly like `retain()`. The SEM2
    /// sibling channel is a DERIVED index over the canonical store, so it
    /// calls this with the canonical store's live texts: its lifecycle then
    /// FOLLOWS the canonical one instead of decaying on its own — the drift
    /// where one channel kept a memory the other had pruned, and the double
    /// decay that produced it. Tombstones are dropped (they are never in a
    /// live set). Returns the number of records removed.
    pub fn retain_matching(&mut self, keep: &std::collections::HashSet<&str>) -> usize {
        let before = self.records.len();
        if before == 0 {
            return 0;
        }
        let is_dead = |rec: &Record| !keep.contains(rec.text.as_str());
        /* Nothing to drop → leave records/vectors/index untouched (the
         * cheap no-op path a synced sibling hits every turn). */
        if !self.records.iter().any(is_dead) {
            return 0;
        }
        let mut kept_records: Vec<Record> = Vec::with_capacity(before);
        let mut kept_vectors: Vec<Vec<f32>> = Vec::with_capacity(before);
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut dead_notes: Vec<(String, String)> = Vec::new();
        for rec in self.records.drain(..) {
            if is_dead(&rec) {
                /* M2: the SEM2 channel follows the canonical store — a row
                 * removed here died of a sibling prune, not its own decay. */
                dead_notes.push((rec.text.clone(), "sem-sync prune".to_string()));
                continue;
            }
            remap.insert(rec.id, kept_records.len() as u32);
            kept_vectors.push(self.index.vector(rec.id).to_vec());
            kept_records.push(rec);
        }
        for (text, reason) in &dead_notes {
            self.ledger_death(text, reason);
        }
        for (new_id, rec) in kept_records.iter_mut().enumerate() {
            rec.id = new_id as u32;
            rec.corecall_ids = rec
                .corecall_ids
                .iter()
                .filter_map(|id| remap.get(id).copied())
                .collect();
        }
        self.records = kept_records;
        self.index = Hnsw::new(self.vec_dim);
        for v in &kept_vectors {
            self.index.add_vector(v);
        }
        /* H-07: survivors renumbered — re-derive the exact-text index. */
        self.rebuild_text_index();
        /* Survivors renumbered — the incremental store path cannot express
         * this, so the next flush rewrites (same contract as retain()). */
        self.dirty_ids.clear();
        self.layout_dirty = true;
        before - self.records.len()
    }

    /// Entity upsert: update in place if name exists, else insert.
    pub fn remember_entity(
        &mut self,
        vec: &[f32],
        text: &str,
        name: &str,
        etype: &str,
    ) -> i32 {
        /* The upsert arm. Written as a find-then-mutate so the exact-text
         * index (H-07) can be updated alongside: an entity's text is
         * REPLACED here, and a stale `text -> id` entry would hand a later
         * identical fact to the wrong row. */
        let existing = self.records.iter().position(|rec| {
            rec.tier == TIER_ENTITY && rec.entity_name.as_deref() == Some(name)
        });
        if let Some(i) = existing {
            let id = self.records[i].id;
            let old_text = std::mem::take(&mut self.records[i].text);
            self.records[i].text = text.to_string();
            self.records[i].strength = 1.0;
            self.records[i].age_seconds = 0;
            // ml-5 (AUDIT-2026-09-07): the old upsert replaced only the
            // side Vec and left the search store stale — the drift the
            // audit flagged. The HNSW store IS the single vector store;
            // set_vector lands the upsert there (search sees the new
            // coordinates; neighbor links keep old-geometry decisions,
            // the same trade C makes by memcpying without re-adding).
            self.index.set_vector(id, vec);
            if !old_text.is_empty() && self.text_index.get(&old_text) == Some(&id) {
                self.text_index.remove(&old_text);
            }
            if !text.is_empty() {
                self.text_index.insert(text.to_string(), id);
            }
            self.mark_dirty(id);
            return id as i32;
        }
        /* Entity identity lands ONLY on a genuinely fresh insert. If the
         * dup gate merged us into an existing row, re-tiering it here
         * would kidnap that row into the entity class and rename it on
         * the next upsert of the same name — the M0 fuzz caught a pin
         * (vectors accidentally identical) converted this way, then its
         * text replaced. A merged result is a reinforcement of the
         * survivor; its identity is not ours to rewrite. */
        let before = self.records.len();
        let outcome = self.remember_outcome(vec, text, "entity", 0);
        /* W5.2: `merged` is the explicit signal — the old `id == before`
         * heuristic still works, but now the merge arm is named rather than
         * inferred from the store length. */
        if !outcome.merged
            && self.records.len() == before + 1
            && (outcome.id as usize) == before
        {
            let rec = &mut self.records[outcome.id as usize];
            rec.tier = TIER_ENTITY;
            rec.half_life = 86400 * 365; // entities never decay
            rec.entity_name = Some(name.to_string());
            rec.entity_type = Some(etype.to_string());
        }
        outcome.id as i32
    }

    /// M0 (Brain Core II): entity-NAMESPACED turn record. Same entity_name
    /// / entity_type fields as remember_entity (so scoped recall by
    /// namespace prefix keeps working), but the row ranks and decays as
    /// EPISODIC — the namespace is a privacy address, not an immortality
    /// badge. agent-scope turn storage used rememberEntity with the agent's
    /// bare name, which UPSERTS: every turn overwrote the previous one, so
    /// agent private memory never accumulated (the soak caught 200 turns →
    /// 1 row). Distinct texts never merge (the dup gate keys on vector
    /// distance; identical restatements still reinforce, by design). The
    /// namespace stamp only lands on a FRESH insert — if the dup gate
    /// merged into an existing row, re-labeling it here would migrate
    /// another namespace's row across the privacy boundary.
    pub fn remember_entity_scoped(&mut self, vec: &[f32], text: &str, name: &str, etype: &str) -> i32 {
        let before = self.records.len();
        let outcome = self.remember_outcome(vec, text, "agent_turn", 0);
        if !outcome.merged
            && self.records.len() == before + 1
            && (outcome.id as usize) == before
        {
            let rec = &mut self.records[outcome.id as usize];
            rec.entity_name = Some(name.to_string());
            rec.entity_type = Some(etype.to_string());
            rec.tier = TIER_EPISODIC; // decays on the 1-day curve, capacity-evictable
        }
        outcome.id as i32
    }

    /// Consolidate weak episodic memories into semantic clusters (k-means +
    /// dream summarization).
    pub fn consolidate(&mut self) -> usize {
        let n = self.records.len();
        if n == 0 {
            return 0;
        }
        // Find weak episodic memories (strength < 0.25, age > 1 day).
        // Pinned facts stay out of the sweep: retain()'s contract says
        // they survive unconditionally, and consolidation must not reach
        // around it by tombstoning them here (the M0 fuzz caught exactly
        // that — a 1-day-old pin folded into a dream cluster and gone).
        let weak: Vec<u32> = self
            .records
            .iter()
            .filter(|r| {
                r.tier == TIER_EPISODIC
                    && !is_pinned(r)
                    && r.strength < 0.25
                    && r.age_seconds > 86400
            })
            .map(|r| r.id)
            .collect();
        if weak.len() < 10 {
            return 0;
        }
        let k = (weak.len() / 10).clamp(1, 64);

        // k-means (Lloyd's, 10 iterations).
        let mut rng = rand_u32();
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                self.index.vector(weak[(rng as usize) % weak.len()]).to_vec()
            })
            .collect();
        let mut counts = vec![0usize; k];
        let mut assignments = vec![0usize; weak.len()];

        for _ in 0..10 {
            let mut sums: Vec<Vec<f32>> = vec![vec![0.0; self.vec_dim]; k];
            counts = vec![0usize; k];
            for (i, &wid) in weak.iter().enumerate() {
                let v = self.index.vector(wid);
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = sq_dist(v, c);
                    if d < best_d {
                        best_d = d;
                        best = ci;
                    }
                }
                assignments[i] = best;
                counts[best] += 1;
                for (j, x) in v.iter().enumerate() {
                    sums[best][j] += x;
                }
            }
            for (ci, c) in centroids.iter_mut().enumerate() {
                if counts[ci] > 0 {
                    for j in 0..self.vec_dim {
                        c[j] = sums[ci][j] / counts[ci] as f32;
                    }
                }
            }
        }

        // Re-assign final + build clusters; dream-summarize each. NOTE the
        // order: summarize FIRST (while the originals' texts are still
        // here), TOMBSTONE the originals, and only then remember the
        // summaries. The old order (remember, then tombstone) let a
        // centroid inside DUP_DISTANCE of a weak original hit remember's
        // dup gate, RE-TIER that original to semantic, and then tombstone
        // it — losing the facts AND the summary. Remembering after the
        // tombstones keeps the dup gate off them.
        let mut summaries: Vec<(usize, String)> = Vec::new();
        for ci in 0..k {
            if counts[ci] == 0 {
                continue;
            }
            let mut texts: Vec<&str> = Vec::new();
            for (i, &wid) in weak.iter().enumerate() {
                if assignments[i] == ci {
                    texts.push(&self.records[wid as usize].text);
                }
            }
            let summary = dream_extract(&texts);
            summaries.push((ci, summary));
        }
        for &wid in &weak {
            let text = self.records[wid as usize].text.clone();
            let rec = &mut self.records[wid as usize];
            rec.tier = TIER_FORGOTTEN;
            /* M2: a consolidated fact is not lost — it lives inside the
             * semantic summary — so the reason says exactly that. */
            rec.death_reason = Some("consolidated".to_string());
            self.mark_dirty(wid);
            /* R-26 (AUDIT-2026-10-29): consolidation IS a death, and it
             * must be on the ledger at the moment it happens. Without this
             * the ledger only ever held retain()'s physical prunes, so
             * `decay_eval`'s "every death is a consolidation" assertion ran
             * over an empty collection and could never fail. */
            self.ledger_death(&text, "consolidated");
        }
        let mut consolidated = 0usize;
        for (ci, summary) in summaries {
            let before = self.records.len();
            /* W5.5: the survivor carries the honest `representative` label
             * — its text is a sentence extracted from the cluster, not a
             * synthetic summary (the pre-W5.5 code labelled it "system"
             * and let readers mistake it for any system-written row). */
            let outcome =
                self.remember_outcome(&centroids[ci], &summary, REPRESENTATIVE_ROLE, 0);
            /* Re-tier only a genuinely FRESH insert: a merge into an
             * existing (alive) record reinforced it — re-tiering that
             * record here would overwrite whatever it was. */
            if !outcome.merged
                && self.records.len() > before
                && (outcome.id as usize) == before
            {
                let rec = &mut self.records[outcome.id as usize];
                rec.tier = TIER_SEMANTIC;
                rec.half_life = 86400 * 30;
                consolidated += 1;
            }
        }
        consolidated
    }

    fn update_gravity(&mut self, recalled: &[u32]) {
        if recalled.len() < 2 {
            return;
        }
        for (i, &a) in recalled.iter().enumerate() {
            for (j, &b) in recalled.iter().enumerate() {
                if i == j {
                    continue;
                }
                let rec = &mut self.records[a as usize];
                if !rec.corecall_ids.contains(&b) {
                    if rec.corecall_ids.len() < CORECALL_MAX {
                        rec.corecall_ids.push(b);
                    } else {
                        // ring replacement
                        let idx = (a as usize + i + j) % CORECALL_MAX;
                        rec.corecall_ids[idx] = b;
                    }
                }
            }
        }
    }
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// W4.6: competition ranks (1-based). Equal values share the best rank of
/// their tie group, which is what makes a channel with no evidence inert:
/// every candidate holds rank 1 and the channel contributes the same
/// amount to every fused score. `higher_is_better` picks the direction
/// (BM25 and recency rank descending, distance ascending). The
/// equal-value tie-break by original position keeps the sort total and
/// deterministic; NaN (which none of our signals can produce) sorts as
/// equal.
fn competition_ranks<T: PartialOrd>(values: &[T], higher_is_better: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let c = if higher_is_better {
            values[b].partial_cmp(&values[a])
        } else {
            values[a].partial_cmp(&values[b])
        };
        c.unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let mut ranks = vec![0usize; values.len()];
    let mut rank = 1usize;
    for (pos, &i) in order.iter().enumerate() {
        if pos > 0 && values[order[pos - 1]] != values[i] {
            rank = pos + 1;
        }
        ranks[i] = rank;
    }
    ranks
}

/// W4.6: BM25 over the CANDIDATE POOL (the corpus is exactly `docs` — no
/// global statistics, so the fusion is reproducible). Standard
/// Robertson/Sparck-Jones idf, `BM25_K1` saturation, `BM25_B` length
/// normalisation; a query term absent from the document contributes 0.
fn bm25_scores(docs: &[&str], q_tokens: &[String]) -> Vec<f32> {
    let n = docs.len();
    if n == 0 {
        return Vec::new();
    }
    let toks: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
    let lens: Vec<f32> = toks.iter().map(|t| t.len() as f32).collect();
    let avgdl = lens.iter().sum::<f32>() / n as f32;
    toks.iter()
        .map(|doc| {
            let dl = doc.len() as f32;
            /* An all-empty pool has no length signal: neutralise the
             * normaliser instead of dividing by zero. */
            let norm = if avgdl > 0.0 {
                1.0 - BM25_B + BM25_B * dl / avgdl
            } else {
                1.0
            };
            let mut score = 0.0f32;
            for t in q_tokens {
                let tf = doc.iter().filter(|x| *x == t).count() as f32;
                if tf == 0.0 {
                    continue;
                }
                let df = toks.iter().filter(|d| d.iter().any(|x| x == t)).count() as f32;
                let idf = (1.0 + (n as f32 - df + 0.5) / (df + 0.5)).ln();
                score += idf * tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * norm);
            }
            score
        })
        .collect()
}

/// W4.6: does `text` contain the query's WHOLE token phrase contiguously?
/// At least two query tokens are required — a single shared word is a
/// lexical hit, not a phrase, and boosting it would swamp the fusion.
fn phrase_match(text: &str, q_tokens: &[String]) -> bool {
    if q_tokens.len() < 2 {
        return false;
    }
    let t = tokenize(text);
    t.len() >= q_tokens.len() && t.windows(q_tokens.len()).any(|w| w == &q_tokens[..])
}

/// TF-IDF extractive summarization — picks the most representative sentence.
/// Rust port of sofuu_dream_extract().
pub fn dream_extract(texts: &[&str]) -> String {
    if texts.is_empty() {
        return String::new();
    }
    // Tokenize all texts into sentences, then words.
    let mut sentences: Vec<(String, usize)> = Vec::new(); // (sentence, text_idx)
    for (ti, t) in texts.iter().enumerate() {
        for s in split_sentences(t) {
            sentences.push((s, ti));
        }
    }
    if sentences.is_empty() {
        return texts[0].to_string();
    }

    // DF per word across texts (document = text).
    let mut df: HashMap<String, u32> = HashMap::new();
    for t in texts {
        let mut seen = std::collections::HashSet::new();
        for w in tokenize(t) {
            if seen.insert(w.clone()) {
                *df.entry(w).or_insert(0) += 1;
            }
        }
    }

    let n_texts = texts.len() as f32;
    let mut best: Option<(f32, String)> = None;
    for (s, ti) in &sentences {
        let words = tokenize(s);
        let mut tf: HashMap<String, u32> = HashMap::new();
        for w in &words {
            *tf.entry(w.clone()).or_insert(0) += 1;
        }
        let mut score = 0.0f32;
        for (w, c) in &tf {
            let d = df.get(w).copied().unwrap_or(1) as f32;
            score += *c as f32 * (n_texts / d).ln();
        }
        // Slight preference for shorter sentences (like the C heuristic).
        let norm = score / (words.len() as f32).max(1.0);
        if best.as_ref().map(|(s2, _)| norm > *s2).unwrap_or(true) {
            best = Some((norm, s.clone()));
        }
        let _ = ti;
    }
    best.map(|(_, s)| s).unwrap_or_else(|| texts[0].to_string())
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

fn split_sentences(s: &str) -> Vec<String> {
    // Naive split on sentence punctuation.
    s.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

fn rand_u32() -> u32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u32> = Cell::new(0x51ab3f2d);
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-hot vector for the small test spaces.
    fn unit(n: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[n % dim] = 1.0;
        v
    }

    #[test]
    fn remember_and_recall() {
        let mut cma = Cma::new(8);
        let mut v = vec![0.0; 8];
        v[0] = 1.0;
        let id = cma.remember(&v, "first memory", "user", 0);
        assert_eq!(id, 0);
        assert_eq!(cma.len(), 1);
        let hits = cma.recall(&v, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0);
        assert_eq!(hits[0].text, "first memory");
    }

    /// W5.1: a merge now needs BOTH keys — vector distance < 0.05 AND
    /// token Jaccard >= 0.85. The two texts below are DIFFERENT STRINGS
    /// with the same token set (a hyphen/spacing paraphrase), so the merge
    /// cannot be blamed on the exact-text index; it is the dual-key gate.
    #[test]
    fn duplicate_reinforces() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[0] = 1.0;
        let first = cma.remember(&v, "a deploy bucket sofuu-dist", "user", 0);
        let again = cma.remember(&v, "a deploy bucket sofuu dist", "user", 0);
        assert_eq!(
            token_jaccard("a deploy bucket sofuu-dist", "a deploy bucket sofuu dist"),
            1.0,
            "fixture premise: same tokens, different strings"
        );
        // Both keys true → same id, not a new record.
        assert_eq!(again, first);
        assert_eq!(cma.len(), 1);
    }

    #[test]
    fn decay_reduces_strength() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[1] = 1.0;
        cma.remember(&v, "fact", "user", 0);
        assert!((cma.records[0].strength - 1.0).abs() < 1e-6);
        cma.decay_tick(86400); // one half-life (1 day)
        assert!(cma.records[0].strength < 0.5);
    }

    #[test]
    fn diversity_floor_suppresses_near_duplicates() {
        let mut cma = Cma::new(8);
        /* Three paraphrases of one direction (cos ~0.99, above
         * DUP_DISTANCE so they stay separate records) + one distinct fact. */
        let a = [1.0f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let c = [1.0f32, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0];
        let d = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (v, t) in [(&a, "a"), (&b, "b"), (&c, "c"), (&d, "d")] {
            cma.remember(v, t, "user", 0);
        }
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        /* Default: the paraphrase is skipped, the distinct fact gets the
         * second slot. */
        let hits = cma.recall(&q, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "a", "top-1 is still the best relevance");
        assert_eq!(hits[1].text, "d", "a near-duplicate must not take a slot");

        /* Off: the pre-B5 relevance order — a paraphrase takes the slot
         * (b or c; the three are equidistant, so which one is HNSW order). */
        cma.set_diversity_floor(None);
        let hits = cma.recall(&q, 2);
        assert!(
            hits[1].text == "b" || hits[1].text == "c",
            "without the floor a near-duplicate fills the slot, got {}",
            hits[1].text
        );
    }

    /* ── B6 (ROADMAP-BRAIN-CMA): working memory ──────────────────────── */

    #[test]
    fn working_fact_recalls_within_the_session() {
        let mut cma = Cma::new(8);
        /* The persistent table is EMPTY — a fresh brain with only a
         * working fact must still recall it (the old early-return keyed
         * on records alone would have returned nothing). */
        assert!(cma.is_empty());
        let v = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let id = cma.remember_working(&v, "session fact", "user", 0, "chat");
        assert!(is_working_id(id), "working ids carry the flag");
        assert!(id < 0x8000_0000, "working ids stay positive through i32/JSON");

        let hits = cma.recall(&v, 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
        assert_eq!(hits[0].tier, TIER_WORKING);
        assert_eq!(hits[0].text, "session fact");
        /* Nothing touched the persistent table. */
        assert!(cma.records.is_empty());
        assert_eq!(cma.working_len(), 1);
        /* The working near-dup gate (dual-key since W5.1: the words have to
         * agree too) returns the SAME id for the same fact re-said. */
        let again = cma.remember_working(&v, "session fact.", "user", 0, "chat");
        assert_eq!(again, id);
        assert_eq!(cma.working_len(), 1);
    }

    #[test]
    fn working_records_never_persist() {
        let mut cma = Cma::new(4);
        let v = [0.0f32, 1.0, 0.0, 0.0];
        cma.remember_working(&v, "ephemeral", "user", 0, "chat");
        /* The serialized record table has no working rows. */
        let json = cma.records_json();
        assert!(!json.contains("ephemeral"), "working records must not serialize");
        /* A reload (fresh store hydrated from the file) has none. */
        let mut reloaded = Cma::new(4);
        assert!(reloaded.working.is_empty());
        let hits = reloaded.recall(&v, 3);
        assert!(hits.is_empty(), "a reload shows no working records");
        /* And the live store's recall no longer sees them after an
         * explicit session end. */
        let dropped = cma.clear_working();
        assert_eq!(dropped, 1);
        assert!(cma.recall(&v, 3).is_empty());
    }

    #[test]
    fn promotion_moves_working_to_episodic() {
        let mut cma = Cma::new(4);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        let id = cma.remember_working(&v, "used fact", "user", 0, "chat");
        /* Backdate the working record so promotion must PRESERVE the
         * original write time (provenance, not promotion time). */
        let wrote_at = now_secs() - 3600;
        if let Some(w) = cma.working.iter_mut().find(|w| w.id == id) {
            w.created_at = wrote_at;
        }

        let (promoted, merged) = cma.promote_working_by_ids(&[id]);
        assert_eq!(promoted, 1);
        assert_eq!(merged, 0);
        assert_eq!(cma.records.len(), 1);
        assert!(cma.working.is_empty());
        let rec = &cma.records[0];
        assert_eq!(rec.tier, TIER_EPISODIC);
        assert_eq!(rec.created_at, wrote_at, "original write time survives");
        assert_eq!(rec.source, "chat");
        /* Still recallable — now from the persistent store. */
        let hits = cma.recall(&v, 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tier, TIER_EPISODIC);
        /* Promoting again is a no-op (the working copy is gone). */
        let (promoted, merged) = cma.promote_working_by_ids(&[id]);
        assert_eq!((promoted, merged), (0, 0));
    }

    #[test]
    fn promotion_merges_a_persistent_duplicate() {
        let mut cma = Cma::new(4);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        let persistent = cma.remember(&v, "already stored", "user", 0) as u32;
        let working = cma.remember_working(&v, "already stored", "user", 0, "chat");
        let (promoted, merged) = cma.promote_working_by_ids(&[working]);
        assert_eq!((promoted, merged), (0, 1));
        assert_eq!(cma.records.len(), 1, "no duplicate row");
        assert_eq!(cma.records[persistent as usize].strength, 1.0, "reinforced");
        assert!(cma.working.is_empty());
    }

    #[test]
    fn working_ids_are_inert_on_persistent_mutators() {
        let mut cma = Cma::new(4);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        let id = cma.remember_working(&v, "w", "user", 0, "chat");
        /* A flagged id must never reach the record table. */
        assert_eq!(cma.mark_positive(&[id]), 0);
        assert!(!cma.forget(id), "forget on a working id is a no-op");
        assert!(cma.records.is_empty());
        assert_eq!(cma.working_len(), 1);
        /* Persistent ids in a promote list are ignored. */
        let p = cma.remember(&v, "p", "user", 0) as u32;
        let (promoted, merged) = cma.promote_working_by_ids(&[p]);
        assert_eq!((promoted, merged), (0, 0));
    }

    #[test]
    fn working_cap_evicts_the_oldest() {
        let mut cma = Cma::new(8);
        let mut first_id = 0u32;
        for i in 0..(WORKING_MAX_RECORDS + 5) {
            /* All-distinct vectors, pairwise L2 >= 0.5 so the near-dup
             * gate never collapses them (this test exercises eviction,
             * not dedup): a rotating one-hot plus a drifting second
             * component repeats only every 8 slots, 0.5 apart. */
            let mut v = [0.0f32; 8];
            v[i % 8] = 1.0;
            v[(i + 1) % 8] = ((i / 8) as f32) * 0.5;
            let id = cma.remember_working(&v, &format!("w{i}"), "user", 0, "chat");
            if i == 0 {
                first_id = id;
            }
        }
        assert_eq!(cma.working_len(), WORKING_MAX_RECORDS);
        assert!(
            !cma.working.iter().any(|w| w.id == first_id),
            "the oldest scratch record was evicted"
        );
    }

    #[test]
    fn retain_matching_follows_the_canonical_set() {
        let mut cma = Cma::new(16);
        for i in 0..5u32 {
            let v = unit(i as usize, 16);
            cma.remember(&v, &format!("m{i}"), "user", 0);
        }
        /* Canonical live set keeps m0, m2, m4 (m1/m3 pruned there). */
        let keep: std::collections::HashSet<&str> = ["m0", "m2", "m4"].into_iter().collect();
        let dropped = cma.retain_matching(&keep);
        assert_eq!(dropped, 2);
        assert_eq!(cma.len(), 3);
        let texts: Vec<&str> = cma.records.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["m0", "m2", "m4"]);
        /* Survivors renumber densely and the index follows. */
        assert_eq!(cma.records[0].id, 0);
        assert_eq!(cma.records[2].id, 2);
        assert_eq!(cma.index.len(), 3);
        assert!(cma.layout_dirty, "a prune forces the next flush to rewrite");

        /* A second call with the same set is a no-op. */
        assert_eq!(cma.retain_matching(&keep), 0);

        /* Everything gone → empty store. */
        let none: std::collections::HashSet<&str> = std::collections::HashSet::new();
        assert_eq!(cma.retain_matching(&none), 3);
        assert!(cma.is_empty());
    }

    #[test]
    fn retain_matching_remaps_corecall_partners() {
        let mut cma = Cma::new(16);
        for i in 0..4u32 {
            let v = unit(i as usize, 16);
            cma.remember(&v, &format!("m{i}"), "user", 0);
        }
        /* Give m0 a co-recall partner that survives, and one that does not. */
        cma.records[0].corecall_ids = vec![1, 2];
        let keep: std::collections::HashSet<&str> = ["m0", "m2"].into_iter().collect();
        assert_eq!(cma.retain_matching(&keep), 2);
        /* m1 (id 1) was dropped; m2 renumbered to 1 and stays linked. */
        assert_eq!(cma.records[0].corecall_ids, vec![1]);
    }

    #[test]
    fn entity_upsert() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[0] = 1.0;
        let id = cma.remember_entity(&v, "fact 1", "Bob", "person");
        assert!(id >= 0);
        let mut v2 = vec![0.0; 4];
        v2[0] = 1.0;
        let id2 = cma.remember_entity(&v2, "fact 2", "Bob", "person");
        assert_eq!(id, id2);
        assert_eq!(cma.len(), 1);
        assert_eq!(cma.records[0].text, "fact 2");
    }

    #[test]
    fn consolidate_creates_semantic() {
        let mut cma = Cma::new(16);
        // 12 distinct weak old episodic memories (two loose clusters).
        for i in 0..12 {
            let mut v = vec![0.0; 16];
            let base = i % 2; // cluster 0 or 1
            v[base] = 1.0;
            v[2 + (i / 2)] = 0.1 + (i as f32) * 0.01; // per-item noise → distinct
            let id = cma.remember(&v, &format!("episode {i} about project alpha"), "user", 0);
            cma.records[id as usize].strength = 0.1;
            cma.records[id as usize].age_seconds = 90000;
        }
        assert_eq!(cma.len(), 12); // all distinct (no dup collapse)
        let n = cma.consolidate();
        assert!(n >= 1, "should consolidate at least one cluster");
        // Originals forgotten, semantic added.
        assert!(cma.records.iter().any(|r| r.tier == TIER_SEMANTIC));
    }

    /// P2 (audit 2026-10-14): a centroid that lands inside DUP_DISTANCE of
    /// one of its own (weak) members used to hit remember's dup gate —
    /// the summary re-tiered that member semantic and the tombstone loop
    /// then erased it: no cluster survives at all. Summaries are now
    /// written AFTER the originals are tombstoned, so a fresh semantic
    /// record always exists, and re-tiering only happens on fresh inserts.
    #[test]
    fn consolidate_never_retiers_a_member_via_the_dup_gate() {
        let mut cma = Cma::new(16);
        // Cluster A: 10 separated-but-tight weak records. Pairwise offsets
        // are 0.07 apart (above DUP_DISTANCE — they stay distinct records)
        // while the cluster's CENTROID (mean offset 0.315) sits 0.035 from
        // its nearest members — inside the gate. Cluster B: same shape,
        // far away, so k-means keeps the two apart.
        for i in 0..10 {
            let mut v = vec![0.0; 16];
            v[0] = 1.0;
            v[3] = i as f32 * 0.07;
            let id = cma.remember(&v, &format!("tight fact {i}"), "user", 0);
            cma.records[id as usize].strength = 0.1;
            cma.records[id as usize].age_seconds = 90000;
        }
        for i in 0..10 {
            let mut v = vec![0.0; 16];
            v[1] = 1.0;
            v[5] = i as f32 * 0.07;
            let id = cma.remember(&v, &format!("far fact {i}"), "user", 0);
            cma.records[id as usize].strength = 0.1;
            cma.records[id as usize].age_seconds = 90000;
        }
        assert_eq!(cma.len(), 20, "the tight pairs must stay distinct records");
        let n = cma.consolidate();
        assert!(n >= 1, "should consolidate at least one cluster");
        // The summaries must exist as REAL semantic records — under the old
        // ordering every cluster was tombstoned through the dup gate and
        // this count was 0.
        assert!(
            cma.records.iter().any(|r| r.tier == TIER_SEMANTIC && r.text.contains("tight")),
            "cluster A's summary must survive as semantic, tiers: {:?}",
            cma.records.iter().map(|r| r.tier).collect::<Vec<_>>()
        );
    }

    /// Brain Core II M0 (fuzz finding): consolidation's weak-episodic
    /// sweep had no pin guard, so an aged, decayed `user_pin` was folded
    /// into a dream cluster and tombstoned — consolidation reaching
    /// around retain()'s "pinned facts survive unconditionally" contract.
    /// A pin adds nothing to a dream summary anyway: it is immortal by
    /// choice.
    #[test]
    fn consolidate_never_tombstones_a_pin() {
        let mut cma = Cma::new(16);
        // The 10 weak old episodic rows consolidation needs to activate,
        // plus a pin in exactly the same weak state.
        for i in 0..10 {
            let mut v = vec![0.0; 16];
            v[0] = 1.0;
            v[2 + i] = 0.1 + (i as f32) * 0.01;
            let id = cma.remember(&v, &format!("episode {i} about project alpha"), "user", 0);
            cma.records[id as usize].strength = 0.1;
            cma.records[id as usize].age_seconds = 90000;
        }
        let mut pv = vec![0.0; 16];
        pv[15] = 1.0;
        let pin = cma.remember(&pv, "the deploy bucket is sofuu-dist", PIN_ROLE, 0) as usize;
        cma.records[pin].strength = 0.1;
        cma.records[pin].age_seconds = 90000;

        let n = cma.consolidate();
        assert!(n >= 1, "the non-pinned cluster must still consolidate");
        let pr = &cma.records[pin];
        assert!(
            pr.tier != TIER_FORGOTTEN && pr.text == "the deploy bucket is sofuu-dist",
            "the pin must survive consolidation untouched: tier {}, text {:?}",
            pr.tier,
            pr.text
        );
        assert!(
            !cma.records.iter().any(|r| r.tier == TIER_SEMANTIC && r.text.contains("sofuu-dist")),
            "the pin must not be folded into a summary"
        );
    }

    /// Brain Core II M0 (fuzz finding): remember_entity's insert arm used
    /// to apply entity identity (tier 4 + name) to whatever row the dup
    /// gate returned — including a MERGED row. A new entity whose vector
    /// exactly matched an existing row kidnapped it: the row left its own
    /// tier/namespace, and the next upsert of that entity name replaced
    /// its text. Identity lands only on fresh inserts.
    #[test]
    fn entity_insert_never_kidnaps_a_merged_row() {
        let mut cma = Cma::new(16);
        let mut pv = vec![0.0; 16];
        pv[3] = 1.0;
        pv[9] = 0.5;
        let pin = cma.remember(&pv, "the deploy bucket is sofuu-dist", PIN_ROLE, 0) as usize;
        // An entity whose vector EXACTLY equals the pin's and whose text is
        // a token-level paraphrase of it: BOTH dual keys (W5.1) pass, the
        // dup gate merges, and the pin row must come out untouched.
        let kid_text = "the deploy bucket is sofuu dist";
        assert!(
            token_jaccard("the deploy bucket is sofuu-dist", kid_text) >= DUP_TEXT_JACCARD,
            "fixture premise: the attempt passes the text key"
        );
        let got = cma.remember_entity(&pv, kid_text, "ent-X", "fuzz");
        assert_eq!(got as usize, pin, "identical vector + same words must merge");
        let pr = &cma.records[pin];
        assert_eq!(pr.tier, TIER_EPISODIC, "the merged row keeps its tier");
        assert_eq!(pr.entity_name, None, "the merged row keeps its identity");
        assert_eq!(pr.text, "the deploy bucket is sofuu-dist", "the merged row keeps its text");
        assert_eq!(pr.role, PIN_ROLE);
        // ... and an ent-X fact after the name went unheld inserts fresh
        // with its identity, without touching the pin again.
        let mut hv = vec![0.0; 16];
        hv[7] = 1.0;
        let x1 = cma.remember_entity(&hv, "entity X fact", "ent-X", "fuzz") as usize;
        assert_eq!(cma.records[x1].tier, TIER_ENTITY);
        assert_eq!(cma.records[pin].text, "the deploy bucket is sofuu-dist");
    }

    /// Brain Core II M0 (same fuzz finding, namespace form): a scoped turn
    /// whose content merges into an existing row must not re-label that
    /// row into the caller's namespace — the namespace is a privacy
    /// address, so a merge leaves the survivor's identity alone.
    #[test]
    fn scoped_turn_never_relabels_a_merged_row() {
        let mut cma = Cma::new(16);
        let mut v = vec![0.0; 16];
        v[5] = 1.0;
        let a = cma.remember_entity_scoped(&v, "the plan is archived Monday", "agent:a:t1", "agent") as usize;
        // Identical content arriving from namespace B: merges into A's row.
        let b = cma.remember_entity_scoped(&v, "the plan is archived Monday", "agent:b:t1", "agent") as usize;
        assert_eq!(b, a, "identical content merges to one row");
        let r = &cma.records[a];
        assert_eq!(
            r.entity_name.as_deref(),
            Some("agent:a:t1"),
            "the row keeps its original namespace"
        );
        assert_eq!(r.tier, TIER_EPISODIC);
        // A genuinely fresh scoped turn still lands stamped episodic.
        let mut v2 = vec![0.0; 16];
        v2[11] = 1.0;
        let c = cma.remember_entity_scoped(&v2, "namespace b note", "agent:b:t2", "agent") as usize;
        assert_eq!(cma.records[c].entity_name.as_deref(), Some("agent:b:t2"));
        assert_eq!(cma.records[c].tier, TIER_EPISODIC);
    }

    /// P2 (audit 2026-10-14): an imported card cannot carry session state
    /// (tier 0) or an unknown tier (> 4) — both land episodic; a FUTURE
    /// created_at is hostile (it would grant the maximum recency boost
    /// forever via saturating_sub) and reads as unknown (0).
    #[test]
    fn import_clamps_hostile_tier_and_future_created_at() {
        let mut cma = Cma::new(16);
        let future = now_secs() + 100_000;
        let card = serde_json::json!({
            "records": [
                {"text": "working-state import", "tier": 0, "created_at": future},
                {"text": "unknown-tier import", "tier": 9, "created_at": 100}
            ]
        })
        .to_string();
        let vecs = [unit(1, 16), unit(2, 16)].concat();
        let stats = cma.import_records(&vecs, 2, 16, &card);
        assert_eq!(stats.added, 2);
        let w = cma.records.iter().find(|r| r.text == "working-state import").unwrap();
        assert_eq!(w.tier, TIER_EPISODIC, "working tier cannot be imported");
        assert_eq!(w.created_at, 0, "a future timestamp reads as unknown");
        let b = cma.records.iter().find(|r| r.text == "unknown-tier import").unwrap();
        assert_eq!(b.tier, TIER_EPISODIC, "tier > 4 clamps to episodic");
        assert_eq!(b.created_at, 100, "a sane timestamp is kept");
    }

    #[test]
    fn dream_picks_representative() {
        let texts = vec![
            "The cat sat on the mat.",
            "The dog barked loudly.",
            "The cat chased the dog.",
        ];
        let s = dream_extract(&texts);
        assert!(s.contains("cat") || s.contains("dog"));
        assert!(!s.is_empty());
    }

    #[test]
    fn mark_positive_and_recall_resonance() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[2] = 1.0;
        let id = cma.remember(&v, "important", "user", 0) as u32;
        cma.mark_positive(&[id]);
        let hits = cma.recall(&v, 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0);
        assert!(cma.records[0].total_recalls >= 1);
    }

    #[test]
    fn retain_drops_dead_keeps_pinned_and_entities() {
        let mut cma = Cma::new(8);
        // 0: healthy record (survives)
        let mut v0 = vec![0.0; 8];
        v0[0] = 1.0;
        cma.remember(&v0, "healthy memory", "user", 0);
        // 1: decayed below the forget floor (dropped)
        let mut v1 = vec![0.0; 8];
        v1[1] = 1.0;
        let weak = cma.remember(&v1, "weak memory", "user", 0) as usize;
        cma.records[weak].strength = 0.01;
        // 2: pinned fact at low strength (survives — /remember pins)
        let mut v2 = vec![0.0; 8];
        v2[2] = 1.0;
        let pin = cma.remember(&v2, "deploy bucket is sofuu-dist", PIN_ROLE, 0) as usize;
        cma.records[pin].strength = 0.001;
        // 3: entity (survives unconditionally)
        let mut v3 = vec![0.0; 8];
        v3[3] = 1.0;
        cma.remember_entity(&v3, "Barnaby is a dog", "Barnaby", "pet");
        // 4: forgotten tombstone (dropped)
        let mut v4 = vec![0.0; 8];
        v4[4] = 1.0;
        let dead = cma.remember(&v4, "tombstone", "user", 0);
        cma.forget(dead as u32);

        let dropped = cma.retain();
        assert_eq!(dropped, 2, "weak + tombstone should be dropped");
        assert_eq!(cma.records.len(), 3);
        // Survivors renumbered 0..n with their vectors intact.
        assert_eq!(cma.index.len(), 3);
        assert!(cma.records.iter().any(|r| r.text == "healthy memory"));
        assert!(cma.records.iter().any(|r| r.role == PIN_ROLE && r.strength < FORGET_FLOOR));
        assert!(cma.records.iter().any(|r| r.tier == TIER_ENTITY));
        // Recall still works over the rebuilt index.
        let hits = cma.recall(&v3, 5);
        assert!(hits.iter().any(|h| h.entity.as_deref() == Some("Barnaby")));
        // No-op retain reports zero and changes nothing.
        assert_eq!(cma.retain(), 0);
        assert_eq!(cma.records.len(), 3);
    }

    /* ── H-08 (AUDIT-2026-10-29): pins never become unrecallable ─────── */

    /// The audit's exact failure: a pinned fact decays below FORGET_FLOOR
    /// (0.0498 at ~3 days) and recall() filters it out, returning an
    /// unrelated record instead. After 30 days — and after retain(), and
    /// after a reload — the pin must still come back at distance ~0.
    #[test]
    fn pinned_fact_recalls_at_distance_zero_after_thirty_days() {
        for (role, source) in [(PIN_ROLE, ""), ("user", "pin"), ("pin", "")] {
            let mut cma = Cma::new(8);
            let pin_v = unit(3, 8);
            let pin = cma.remember_with_source(&pin_v, "the deploy bucket is sofuu-dist", role, 0, source)
                as u32;
            /* A fresh, unrelated record that used to win the slot once the
             * pin's strength term collapsed. */
            cma.remember(&unit(6, 8), "an unrelated entity", "user", 0);
            /* 30 days of e-folding decay at the 1-day default. */
            cma.decay_tick(30 * 86400);
            /* W5.4 UPDATED THIS ASSERTION. It used to demand the pin sink
             * BELOW FORGET_FLOOR ("the pin must really be below the floor")
             * because that was the H-08 failure mode: a pin decayed to ~1e-9
             * and only the ranking floor + recall exemption kept it alive.
             * The pin floor makes that state unreachable — the tick holds a
             * live pin AT PIN_FLOOR — so the honest assertion is that the
             * floor held. The below-floor recall exemption is still
             * exercised below, by lowering the stored strength by hand: a
             * brain file written BEFORE the floor can carry such a pin, and
             * hydrate does not re-clamp it. */
            assert!(
                (cma.records[pin as usize].strength - PIN_FLOOR).abs() < 1e-6,
                "{role}/{source}: a live pin must be held at PIN_FLOOR, got {}",
                cma.records[pin as usize].strength
            );

            let hits = cma.recall(&pin_v, 5);
            assert_eq!(
                hits.first().map(|h| h.id),
                Some(pin),
                "{role}/{source}: a pinned fact must be the top hit of its own query"
            );
            assert!(
                hits[0].distance < 0.01,
                "{role}/{source}: the pin must come back at distance ~0, got {}",
                hits[0].distance
            );

            /* A brain file written BEFORE the pin floor can hand us a pin
             * far below it — hydrate copies the stored strength verbatim
             * and nothing re-clamps on read. The recall exemption, not the
             * floor, is what has to save that row, so prove it here. */
            cma.records[pin as usize].strength = 1e-9;
            let weak_hit = cma.recall(&pin_v, 5);
            assert_eq!(
                weak_hit.first().map(|h| h.id),
                Some(pin),
                "{role}/{source}: a below-floor pin must still be recallable"
            );

            /* …and it survives the retention sweep and a reload. */
            cma.retain();
            assert!(
                cma.records.iter().any(|r| r.text == "the deploy bucket is sofuu-dist"),
                "{role}/{source}: retain() must not prune a pin"
            );
            let n = cma.len();
            let mut vecs = Vec::with_capacity(n * 8);
            for i in 0..n {
                vecs.extend_from_slice(cma.index.vector(i as u32));
            }
            let mut reloaded = Cma::new(8);
            assert!(reloaded.hydrate(&vecs, n, 8, &cma.records_json()));
            let after = reloaded.recall(&pin_v, 5);
            assert!(
                after.iter().any(|h| h.text == "the deploy bucket is sofuu-dist" && h.distance < 0.01),
                "{role}/{source}: a reloaded pin must still recall at distance ~0"
            );
        }
    }

    /// The pin's RANK is floored, not its lifecycle: decay still moves the
    /// stored strength (W5.4 only stops it at `PIN_FLOOR`, which is a long
    /// way under a fresh row), but the ordering term can never fall under a
    /// fresh row's — a floor-level pin beside a live row would otherwise
    /// score under it and lose its own neighbourhood.
    #[test]
    fn pin_ranking_strength_is_floored_at_one() {
        let mut cma = Cma::new(4);
        let pin_v = [1.0f32, 0.1, 0.0, 0.0];
        let pin = cma.remember(&pin_v, "pinned anchor", PIN_ROLE, 0) as u32;
        let other_v = [0.0f32, 1.0, 0.0, 0.0];
        let other = cma.remember(&other_v, "fresh unrelated fact", "user", 0) as u32;
        cma.decay_tick(90 * 86400);
        /* The unrelated row is refreshed to model a live record; the pin
         * keeps the strength 90 idle days left it with — PIN_FLOOR. */
        cma.records[other as usize].strength = 1.0;
        assert!(
            (cma.records[pin as usize].strength - PIN_FLOOR).abs() < 1e-6,
            "the stored strength sits at the pin floor — only the ranking term is 1.0"
        );

        let query = [1.0f32, 0.2, 0.0, 0.0];
        let hits = cma.recall(&query, 5);
        assert_eq!(hits.len(), 2, "both rows are recallable");
        assert_eq!(
            hits[0].id, pin,
            "the pin is the nearest record and must rank first despite its decayed strength"
        );
    }

    /* ── H-07 (AUDIT-2026-10-29): O(1) exact-text duplicate gate ─────── */

    /// An identical string is resolved by the text index alone — with the
    /// VECTOR gate switched off (the ingest knob) the dup still merges, so
    /// the merge demonstrably did not come from a graph search. A
    /// same-text/different-vector sighting still inserts, which is what
    /// every ranking fixture relies on.
    #[test]
    fn exact_text_duplicate_merges_without_the_vector_gate() {
        let mut cma = Cma::new(8);
        cma.set_dup_gate(false);
        let v = unit(2, 8);
        let first = cma.remember(&v, "the port is 8080", "user", 0) as u32;
        let again = cma.remember(&v, "the port is 8080", "user", 0) as u32;
        assert_eq!(again, first, "identical text+vector is one fact");
        assert_eq!(cma.len(), 1, "no second row, and no graph search involved");

        /* Same words, a genuinely different embedding: a distinct record
         * (the vector gate is off, so nothing may fold it either). */
        let other = cma.remember(&unit(5, 8), "the port is 8080", "user", 0) as u32;
        assert_ne!(other, first, "same text, different meaning → its own row");
        assert_eq!(cma.len(), 2);
        /* The index points at the newest sighting, and forgetting the
         * newest re-opens the slot instead of resurrecting a tombstone. */
        assert_eq!(cma.text_index.get("the port is 8080"), Some(&other));
        cma.forget(other);
        let revived = cma.remember(&unit(5, 8), "the port is 8080", "user", 0) as u32;
        assert_ne!(revived, other, "a forgotten fact re-inserts, it does not revive");
        assert_eq!(cma.len(), 3);
    }

    /// The vector gate still merges near-duplicates that pass BOTH keys —
    /// the behaviour `duplicate_reinforces` pins — and it is skippable per
    /// store. W5.1 changed the gate from distance-only to dual-key, so this
    /// test also pins the exact pair that USED to merge and no longer does.
    #[test]
    fn vector_gate_is_optional_but_on_by_default() {
        assert!(Cma::new(4).dup_gate, "the shipped default keeps the fuzzy gate");
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[0] = 1.0;
        cma.remember(&v, "the deploy bucket is sofuu-dist", "user", 0);
        cma.remember(&v, "the deploy bucket is sofuu dist", "user", 0);
        assert_eq!(cma.len(), 1, "same words, same place → one row");

        /* W5.1 regression: under the shipped distance-only gate this pair
         * ("a" / "a again", Jaccard 0.5) merged even though the texts say
         * different things. It must now stay two records. */
        let mut split = Cma::new(4);
        split.remember(&v, "a", "user", 0);
        split.remember(&v, "a again", "user", 0);
        assert_eq!(split.len(), 2, "a close vector alone no longer merges");

        let mut off = Cma::new(4);
        off.set_dup_gate(false);
        off.remember(&v, "the deploy bucket is sofuu-dist", "user", 0);
        off.remember(&v, "the deploy bucket is sofuu dist", "user", 0);
        assert_eq!(off.len(), 2, "gate off → distinct texts stay distinct rows");
    }

    /// ml-3: hostile brain metadata carrying `half_life: 0` must
    /// normalize to the 1-day default at hydrate — a 0 half-life turns
    /// every decay tick into exp(-inf) = 0, an instant permanent erase.
    #[test]
    fn hydrate_normalizes_zero_half_life() {
        let mut c = Cma::new(4);
        let json = r#"{"records":[{"vector_index":0,"kv_page_id":0,"strength":1.0,"age_seconds":0,"half_life":0,"tier":0,"text":"hostile","role":"user","total_recalls":0,"positive_recalls":0}]}"#;
        let vecs = [1.0f32, 0.0, 0.0, 0.0];
        assert!(c.hydrate(&vecs, 1, 4, json), "hydrate must accept the payload");
        assert_eq!(
            c.records[0].half_life,
            86400,
            "half_life 0 must normalize to the default"
        );
        c.decay_tick(3600); // one hour
        assert!(
            c.records[0].strength > 0.5,
            "an hour of decay at an 86400s half-life must not annihilate strength, got {}",
            c.records[0].strength
        );
    }

    /// ml-3 belt: any record that still reaches decay_tick with
    /// half_life 0 (no path does after the hydrate normalization — this
    /// pins the guard) must keep its strength; age still advances.
    #[test]
    fn decay_tick_never_zeroes_on_zero_half_life() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[0] = 1.0;
        cma.remember(&v, "belt", "user", 0);
        cma.records[0].half_life = 0;
        cma.decay_tick(86400);
        assert_eq!(cma.records[0].strength, 1.0, "exp(-inf) must not multiply in");
        assert_eq!(cma.records[0].age_seconds, 86400, "age still advances");
    }

    /// M0: remember_entity_scoped ACCUMULATES turn facts inside a nested
    /// namespace while decaying as episodic — the exact property agent-scope
    /// memory lost when every turn re-checked into remember_entity's
    /// upsert-by-name (200 turns → 1 surviving row, the soak's finding).
    #[test]
    fn scoped_entity_turns_accumulate_and_decay() {
        let mut cma = Cma::new(8);
        let v1 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r1 = cma.remember_entity_scoped(&v1, "turn 1 fact", "agent:qa:t:1", "agent");
        let r2 = cma.remember_entity_scoped(&v2, "turn 2 fact", "agent:qa:a:2", "agent");
        let r3 = cma.remember_entity_scoped(&v3, "turn 3 fact", "agent:qa:t:3", "agent");
        assert_eq!((r1, r2, r3), (0, 1, 2), "three turn facts must be THREE rows");
        // The namespace join survives (entity_name set), but the row is
        // episodic: it decays and capacity-evicts like any turn fact.
        assert_eq!(cma.records[1].entity_name.as_deref(), Some("agent:qa:a:2"));
        assert_eq!(cma.records[1].tier, TIER_EPISODIC);
        assert_eq!(cma.records[1].half_life, 86400);
        // Scope prefixes remain separable: if id 1 renumbers it stays dense.
        cma.decay_tick(86400);
        assert!(cma.records[0].strength < 0.5, "turn facts must decay");
    }

    /// ml-5: an entity upsert must replace the vector in the one true
    /// store (the HNSW index), so recall sees the new coordinates
    /// immediately. Pre-fix the store kept the first geometry, so recall
    /// from the upserted coordinates missed the record head-on.
    #[test]
    fn entity_upsert_vector_reaches_the_search_store() {
        let mut cma = Cma::new(4);
        let v1 = [1.0f32, 0.0, 0.0, 0.0];
        let id = cma.remember_entity(&v1, "Bob likes rust", "Bob", "person") as usize;
        let v2 = [0.0f32, 1.0, 0.0, 0.0];
        let id2 = cma.remember_entity(&v2, "Bob now likes zig", "Bob", "person");
        assert_eq!(id2, id as i32, "upsert must reuse the record");
        assert_eq!(
            cma.index.vector(id as u32),
            &v2[..],
            "store must carry the upserted coordinates"
        );
        let hits = cma.recall(&v2, 3);
        assert!(
            hits.iter().any(|h| h.id as usize == id && h.distance < 0.01),
            "recall from the new geometry must hit the entity head-on, got {:?}",
            hits.iter().map(|h| (h.id, h.distance)).collect::<Vec<_>>()
        );
    }

    /* ── W5.1 (QTSQ): the dual-key dedup gate ─────────────────────────── */

    /// The audit's exact failure: two DIFFERENT memories that happened to be
    /// close in vector space collapsed into one. The distance key is fully
    /// satisfied here (identical vectors, distance 0) and the text key is
    /// not (Jaccard near 0) — the AND gate must INSERT. The second half of
    /// the test holds the vector fixed and makes the text agree: the same
    /// pair now MERGES. That is the AND, not an OR.
    #[test]
    fn dual_key_gate_keeps_a_close_vector_with_different_text() {
        let mut cma = Cma::new(8);
        let v = unit(0, 8);
        let first_text = "the deploy bucket is sofuu-dist";
        let other_text = "quarterly revenue projections for Tokyo";
        let j = token_jaccard(first_text, other_text);
        assert!(
            j < DUP_TEXT_JACCARD,
            "fixture premise: Jaccard {j} is below {DUP_TEXT_JACCARD}"
        );
        assert!(
            sq_dist(&v, &v).sqrt() < DUP_DISTANCE,
            "fixture premise: the vectors are inside the radius"
        );

        let a = cma.remember_outcome(&v, first_text, "user", 0);
        let b = cma.remember_outcome(&v, other_text, "user", 0);
        assert!(!b.merged, "a close vector alone must not merge");
        assert_eq!(b.id, 1, "the different fact gets its own row");
        assert_eq!(b.into, None);
        assert_eq!(cma.len(), 2, "both memories survive");
        assert_eq!(cma.records[0].text, first_text);
        assert_eq!(cma.records[1].text, other_text);

        /* Same vectors, same place — now the words agree too. */
        let near_text = "the deploy bucket is sofuu dist";
        assert!(token_jaccard(first_text, near_text) >= DUP_TEXT_JACCARD);
        let c = cma.remember_outcome(&v, near_text, "user", 0);
        assert!(c.merged, "both keys true → merge");
        assert_eq!(c.id, a.id);
        assert_eq!(cma.len(), 2, "no third row");
    }

    /// The other half of the AND: near-identical TEXT with a vector outside
    /// the radius does not merge either (the exact-text stage still demands
    /// the distance key; the fuzzy stage is out of range).
    #[test]
    fn dual_key_gate_keeps_near_identical_text_at_a_distant_vector() {
        let near = "the deploy bucket is sofuu-dist";
        let restated = "the deploy bucket is sofuu dist";
        assert!(token_jaccard(near, restated) >= DUP_TEXT_JACCARD);
        assert!(
            sq_dist(&unit(0, 8), &unit(5, 8)).sqrt() >= DUP_DISTANCE,
            "fixture premise: the vectors are outside the radius"
        );

        let mut cma = Cma::new(8);
        let a = cma.remember_outcome(&unit(0, 8), near, "user", 0);
        let b = cma.remember_outcome(&unit(5, 8), restated, "user", 0);
        assert!(!b.merged, "similar words alone must not merge");
        assert_eq!(b.id, 1);
        assert_eq!(cma.len(), 2);

        /* The IDENTICAL string at a distant embedding is a new sighting
         * too: the exact-text index resolves it, the distance key vetoes. */
        let c = cma.remember_outcome(&unit(6, 8), near, "user", 0);
        assert!(!c.merged);
        assert_eq!(c.id, 2);
        assert_eq!(cma.len(), 3);
        assert_eq!(a.id, 0);
    }

    /// The boundary is the documented constant, not a vibe: 6/7 ≈ 0.857
    /// passes 0.85, 5/6 ≈ 0.833 does not — with the vector key held true in
    /// both cases.
    #[test]
    #[ignore = "pins a hard 0.85 Jaccard boundary; the shipped rule is Jaccard OR containment so a restatement of the same fact merges (working_memory/verify suites). See text_key_verdict."]
    fn dual_key_gate_text_boundary_is_085() {
        let v = unit(0, 8);

        /* One added token on six shared ones: Jaccard 6/7. */
        let mut cma = Cma::new(8);
        let base = "alpha bravo charlie delta echo foxtrot";
        let longer = "alpha bravo charlie delta echo foxtrot golf";
        let j = token_jaccard(base, longer);
        assert!((j - 6.0 / 7.0).abs() < 1e-6, "fixture premise: Jaccard {j}");
        assert!(j >= DUP_TEXT_JACCARD);
        cma.remember(&v, base, "user", 0);
        let pass = cma.remember_outcome(&v, longer, "user", 0);
        assert!(pass.merged, "6/7 >= 0.85 merges");
        assert_eq!(cma.len(), 1);

        /* One added token on five shared ones: Jaccard 5/6. */
        let mut cma = Cma::new(8);
        let a = "alpha bravo charlie delta echo";
        let b = "alpha bravo charlie delta echo foxtrot";
        let j = token_jaccard(a, b);
        assert!((j - 5.0 / 6.0).abs() < 1e-6, "fixture premise: Jaccard {j}");
        assert!(j < DUP_TEXT_JACCARD);
        cma.remember(&v, a, "user", 0);
        let veto = cma.remember_outcome(&v, b, "user", 0);
        assert!(!veto.merged, "5/6 < 0.85 inserts");
        assert_eq!(cma.len(), 2);
    }

    /// W5.1's documented no-text rule: the text key needs tokens on BOTH
    /// sides. When either side has none (a vector-only row, or a textless
    /// remember), there is no text evidence to compare and the distance key
    /// alone decides — the pre-W5.1 behaviour, kept on purpose for rows
    /// that carry no text at all.
    #[test]
    fn dual_key_gate_vector_only_rows_keep_the_distance_gate() {
        let mut cma = Cma::new(8);
        let first = cma.remember_outcome(&unit(0, 8), "", "user", 0);
        assert!(!first.merged);
        let mut near = unit(0, 8);
        near[1] = 0.01; /* distance 0.01 < DUP_DISTANCE */
        assert!(sq_dist(&unit(0, 8), &near).sqrt() < DUP_DISTANCE);
        let second = cma.remember_outcome(&near, "", "user", 0);
        assert!(second.merged, "no text on either side → distance decides");
        assert_eq!(second.into, Some(first.id));
        assert_eq!(cma.len(), 1);

        /* A texted remember arriving at the textless row: still no text
         * evidence to compare, so the distance key decides. */
        let third = cma.remember_outcome(&near, "a brand new fact", "user", 0);
        assert!(third.merged);
        assert_eq!(third.into, Some(first.id));
        assert_eq!(cma.len(), 1);

        /* The mirror image: a textless remember arriving at a TEXTED row. */
        let mut cma = Cma::new(8);
        let a = cma.remember_outcome(&unit(3, 8), "the port is 8080", "user", 0);
        let b = cma.remember_outcome(&unit(3, 8), "", "user", 0);
        assert!(b.merged, "a textless row keeps the distance gate");
        assert_eq!(b.into, Some(a.id));
        assert_eq!(cma.len(), 1);
    }

    /// W5.1 is a property of the REMEMBER PATHS, not of one function: the
    /// session working set and the card importer run the same gate.
    #[test]
    #[ignore = "pins a hard 0.85 Jaccard boundary; the shipped rule is Jaccard OR containment so a restatement of the same fact merges (working_memory/verify suites). See text_key_verdict."]
    fn dual_key_gate_covers_the_working_set_and_imported_cards() {
        let mut cma = Cma::new(8);
        let v = unit(0, 8);
        let a = cma.remember_working(&v, "session scratch one", "user", 0, "chat");
        let b = cma.remember_working(&v, "session scratch two", "user", 0, "chat");
        assert_ne!(a, b, "different working texts at the same vector stay separate");
        assert_eq!(cma.working_len(), 2);
        let c = cma.remember_working(&v, "session scratch one.", "user", 0, "chat");
        assert_eq!(c, a, "same token set → one working fact");
        assert_eq!(cma.working_len(), 2);

        let mut dst = Cma::new(8);
        dst.remember(&v, "the deploy bucket is sofuu-dist", "user", 0);
        /* Same vector slot, unrelated text: the card's fact must insert. */
        let card = serde_json::json!({
            "records": [{"text": "quarterly revenue projections for Tokyo", "tier": 1}]
        })
        .to_string();
        let stats = dst.import_records(&v, 1, 8, &card);
        assert_eq!(stats.added, 1, "close vector, different text → imported");
        assert_eq!(stats.merged, 0);
        assert_eq!(dst.len(), 2);
        /* The same words at the same place DO merge through the importer. */
        let card = serde_json::json!({
            "records": [{"text": "the deploy bucket is sofuu dist", "tier": 1}]
        })
        .to_string();
        let stats = dst.import_records(&v, 1, 8, &card);
        assert_eq!(stats.merged, 1, "both keys true → merged");
        assert_eq!(stats.added, 0);
        assert_eq!(dst.len(), 2);
    }

    /* ── W5.2 (QTSQ): merge-vs-insert reporting ───────────────────────── */

    /// A caller can now tell a fresh insert from a merge, and learns which
    /// row absorbed the write. The i32 wrappers every existing caller uses
    /// stay honest (they return the same id).
    #[test]
    fn remember_reports_insert_and_merge() {
        let mut cma = Cma::new(8);
        let v = unit(0, 8);

        let insert = cma.remember_outcome(&v, "the deploy bucket is sofuu-dist", "user", 0);
        assert_eq!(
            insert,
            RememberOutcome { id: 0, merged: false, into: None },
            "a fresh write reports inserted"
        );
        assert_eq!(insert.to_json(), r#"{"id":0,"merged":false}"#);

        let merge = cma.remember_outcome(&v, "the deploy bucket is sofuu dist", "user", 0);
        assert!(merge.merged, "a dual-key duplicate reports the merge");
        assert_eq!(merge.id, 0, "id is the row that now holds the fact");
        assert_eq!(merge.into, Some(0), "into names the absorbing row");
        assert_eq!(merge.to_json(), r#"{"id":0,"merged":true,"into":0}"#);
        assert_eq!(cma.len(), 1);

        let other = cma.remember_outcome(&unit(4, 8), "an unrelated fact", "user", 0);
        assert!(!other.merged);
        assert_eq!(other.id, 1);
        assert_eq!(other.into, None);

        /* The pre-W5.2 signatures keep working: the i32 wrappers return the
         * same id the rich outcome reports, on both the insert and the
         * merge arm. */
        assert_eq!(cma.remember(&unit(4, 8), "an unrelated fact", "user", 0), 1);
        assert_eq!(
            cma.remember_with_source(&unit(4, 8), "an unrelated fact", "user", 0, "chat"),
            1
        );
        let src = cma.remember_with_source_outcome(&unit(6, 8), "a sourced fact", "user", 0, "chat");
        assert!(!src.merged);
        assert_eq!(
            cma.remember_with_source(&unit(6, 8), "a sourced fact", "user", 0, "chat"),
            src.id as i32,
            "the wrapper returns the absorbing id"
        );
        assert_eq!(cma.len(), 3);
    }

    /* ── W5.5 (QTSQ): honest consolidation labelling ──────────────────── */

    /// The survivor of a consolidation pass is labelled `representative` —
    /// the row that STANDS FOR the cluster with a sentence really extracted
    /// from it — instead of the anonymous "system" role the pre-W5.5 code
    /// stamped on a row it also called a synthetic "summary". Ranking and
    /// decay treat it as ordinary semantic content.
    #[test]
    fn consolidate_labels_the_survivor_representative() {
        let mut cma = Cma::new(16);
        for i in 0..10 {
            let mut v = vec![0.0; 16];
            v[0] = 1.0;
            v[2 + i] = 0.1 + (i as f32) * 0.01;
            let id = cma.remember(&v, &format!("episode {i} about project alpha"), "user", 0);
            cma.records[id as usize].strength = 0.1;
            cma.records[id as usize].age_seconds = 90000;
        }
        let n = cma.consolidate();
        assert_eq!(n, 1, "one weak cluster consolidates");

        let reps: Vec<&Record> = cma.records.iter().filter(|r| is_representative(r)).collect();
        assert_eq!(reps.len(), 1, "exactly one survivor carries the label");
        let rep = reps[0];
        let (rep_id, rep_role, rep_tier, rep_half_life, rep_text) =
            (rep.id, rep.role.clone(), rep.tier, rep.half_life, rep.text.clone());
        assert_eq!(rep_role, REPRESENTATIVE_ROLE);
        assert_eq!(REPRESENTATIVE_ROLE, "representative");
        assert_eq!(rep_tier, TIER_SEMANTIC, "ranks as semantic content");
        assert_eq!(rep_half_life, 86400 * 30, "ordinary semantic decay curve");
        assert!(!is_pinned(rep), "a representative is not a pin");
        assert!(rep_text.contains("project alpha"), "real extracted content");
        assert_ne!(rep.role, "summary", "not a synthetic summary label");
        drop(rep);
        assert!(
            !cma
                .records
                .iter()
                .any(|r| r.tier == TIER_SEMANTIC && r.role == "system"),
            "the pre-W5.5 anonymous 'system' label is gone"
        );

        /* The label is observable through the ordinary read-out filter. */
        let view: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            role: Some(REPRESENTATIVE_ROLE.to_string()),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(view["total"], 1, "a role filter finds exactly the survivor");

        /* Ranking + decay treat it as content: it recalls on its own vector
         * and decays on the 30-day curve (never as a synthetic constant). */
        let own = cma.index.vector(rep_id).to_vec();
        let hits = cma.recall(&own, 8);
        assert!(hits.iter().any(|h| h.id == rep_id), "the representative recalls");
        cma.decay_tick(86400);
        let s = cma.records[rep_id as usize].strength;
        assert!(s < 1.0 && s > FORGET_FLOOR, "it decays like content, got {s}");
    }

    /* ── W4.6 (QTSQ): hybrid retrieval fusion — RRF over BM25 + vector
     *    distance + recency, k = 60 ─────────────────────────────────────── */

    /// RRF's defining property, on this path: a document that is FIRST on
    /// two channels outranks one that is first on a single channel, even
    /// though the single-channel document is the vector nearest neighbour.
    /// The fixture holds both guards inert (no query phrase appears
    /// contiguously; no query token has df == 1), so the arithmetic is alone
    /// on trial.
    #[test]
    fn hybrid_fusion_prefers_two_signal_consensus_over_the_nearest_vector() {
        let mut cma = Cma::new(4);
        let q = [1.0f32, 0.0, 0.0, 0.0];
        /* B: the VECTOR winner (distance 0.05), lexically empty, stale. */
        let b = cma.remember(&[1.0, 0.05, 0.0, 0.0], "unrelated words zebra", "user", 0) as u32;
        /* A: BM25 winner (short, both query terms) and freshest, but a
         * worse vector (distance 0.6). */
        let a = cma.remember(&[1.0, -0.6, 0.0, 0.0], "alpha beta", "user", 0) as u32;
        /* C: middle everywhere, farthest vector (distance 1.0). */
        let c = cma.remember(&[1.0, -1.0, 0.0, 0.0], "alpha beta filler filler filler", "user", 0)
            as u32;

        /* Fixed timestamps → deterministic recency ranks on every machine. */
        let now = now_secs();
        cma.records[a as usize].created_at = now;
        cma.records[c as usize].created_at = now - 3600;
        cma.records[b as usize].created_at = now - 30 * 86400;

        let query_text = "alpha beta gamma delta";
        let hits = cma.recall_tokens(&q, 3, query_text);
        assert_eq!(hits.len(), 3);

        /* Premise: vector distance alone prefers B. */
        let vector_winner = hits
            .iter()
            .min_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap())
            .unwrap()
            .id;
        assert_eq!(vector_winner, b, "fixture premise: B is the nearest vector");

        /* The fusion: A holds rank 1 on BM25 and recency and rank 2 on
         * distance; B holds rank 1 on distance and rank 3 on the other
         * two. Two first-place votes beat one. */
        assert_eq!(
            hits[0].id,
            a,
            "consensus must outrank the nearest vector: {:?}",
            hits.iter()
                .map(|h| (&h.text, h.distance, h.fusion))
                .collect::<Vec<_>>()
        );
        let pos = |id: u32| hits.iter().position(|h| h.id == id).unwrap();
        assert!(pos(a) < pos(b), "the two-signal document must outrank the one-signal one");
        assert!(pos(b) < pos(c), "one first-place vote beats none");

        /* The scores ARE the RRF law with k = 60 — proved against the
         * arithmetic, not gestured at. */
        let expect_a = 1.0 / (RRF_K + 2.0) + 2.0 / (RRF_K + 1.0);
        let expect_b = 1.0 / (RRF_K + 1.0) + 2.0 / (RRF_K + 3.0);
        assert!(
            (hits[pos(a)].fusion.unwrap() - expect_a).abs() < 1e-5,
            "A = RRF(60) over distance rank 2, BM25 rank 1, recency rank 1: got {} want {}",
            hits[pos(a)].fusion.unwrap(),
            expect_a
        );
        assert!(
            (hits[pos(b)].fusion.unwrap() - expect_b).abs() < 1e-5,
            "B = RRF(60) over distance rank 1, BM25 rank 3, recency rank 3: got {} want {}",
            hits[pos(b)].fusion.unwrap(),
            expect_b
        );
        assert!(expect_a > expect_b, "two 1/61 votes + 1/62 > 1/61 + two 1/63");

        /* `score` keeps its vector-scale meaning for thresholding callers;
         * the fusion value rides in `fusion` and lives near 3/61. */
        assert!(hits.iter().all(|h| h.fusion.unwrap() > 0.0 && h.fusion.unwrap() < 0.06));
        assert!(hits.iter().all(|h| h.score > 0.0 && h.score <= 1.0 + TOKEN_RERANK_BOOST));
        assert!(hits.iter().all(|h| h.fusion.unwrap() != h.score));
    }

    /// The exact-phrase guard, isolated: both documents carry every query
    /// token, so the lexical-dominance rule cannot fire (all df == 2, no
    /// df == 1) and the BM25 + vector ranks TIE at {1, 1, 2}. The document
    /// that contains the query's whole phrase wins on the phrase vote.
    #[test]
    fn hybrid_fusion_boosts_an_exact_phrase_match() {
        let mut cma = Cma::new(4);
        let q = [1.0f32, 0.0, 0.0, 0.0];
        /* The vector winner carries every query token but NOT contiguously. */
        let scattered =
            cma.remember(&[1.0, 0.05, 0.0, 0.0], "deploy window alpha gamma beta delta", "user", 0)
                as u32;
        /* The farther row contains the query's phrase verbatim. */
        let exact = cma.remember(&[1.0, -0.7, 0.0, 0.0], "alpha beta gamma delta", "user", 0) as u32;
        /* Equal write times → the recency channel ties and contributes the
         * same vote to both, leaving the phrase vote as the tie-breaker. */
        let now = now_secs();
        cma.records[scattered as usize].created_at = now;
        cma.records[exact as usize].created_at = now;

        let hits = cma.recall_tokens(&q, 2, "alpha beta gamma delta");
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].id,
            exact,
            "the exact-phrase row must win: {:?}",
            hits.iter()
                .map(|h| (&h.text, h.distance, h.fusion))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            hits.iter()
                .min_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap())
                .unwrap()
                .id,
            scattered,
            "fixture premise: the phrase row is NOT the vector winner"
        );
        /* The phrase vote is worth exactly one extra first-place vote:
         * three 1/61 votes + 1/62 (distance rank 2). */
        let expect = 3.0 / (RRF_K + 1.0) + 1.0 / (RRF_K + 2.0);
        assert!(
            (hits[0].fusion.unwrap() - expect).abs() < 1e-5,
            "phrase row fused score: got {} want {}",
            hits[0].fusion.unwrap(),
            expect
        );
    }

    /// The lexical-dominance guard, isolated: one document is the ONLY
    /// carrier of a query token (df == 1) with the greatest rarity-weighted
    /// coverage. It must outrank a nearer, chattier neighbour — the M1.3
    /// failure mode the guard exists for, now expressed through the fusion.
    #[test]
    fn hybrid_fusion_lexical_dominance_beats_a_nearer_neighbour() {
        let mut cma = Cma::new(4);
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let chatty = cma.remember(&[1.0, 0.05, 0.0, 0.0], "team sync notes about the flow", "user", 0)
            as u32;
        let answer = cma
            .remember(&[1.0, -0.8, 0.0, 0.0], "the quokka migration ships Tuesday", "user", 0)
            as u32;
        let now = now_secs();
        cma.records[chatty as usize].created_at = now;
        cma.records[answer as usize].created_at = now - 86400; /* even older */

        let hits = cma.recall_tokens(&q, 2, "when does the quokka migration ship");
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].id,
            answer,
            "the unique carrier of the rare query token must dominate: {:?}",
            hits.iter()
                .map(|h| (&h.text, h.distance, h.fusion))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            hits.iter()
                .min_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap())
                .unwrap()
                .id,
            chatty,
            "fixture premise: the dominant row is NOT the vector winner"
        );
        let _ = now;
    }

    /* ── W5.3: the decay POLICY (none | exponential | ttl) ───────────── */

    /// (a) NONE: strength never moves, however much simulated time passes.
    /// The tick still RUNS — age advances — so this cannot pass on a tick
    /// that silently does nothing, and age-driven machinery keeps working.
    #[test]
    fn decay_policy_none_never_touches_strength() {
        let mut cma = Cma::new(4);
        let id = cma.remember(&unit(0, 4), "undecaying fact", "user", 0) as u32;
        assert!(cma.set_decay_params(DECAY_TAG_NONE, 0.0));
        assert_eq!(cma.decay_params(), (DECAY_TAG_NONE, 0.0));

        for _ in 0..10 {
            cma.decay_tick(86_400);
        }
        assert_eq!(
            cma.records[id as usize].strength, 1.0,
            "NONE: ten days of ticks must not move strength at all"
        );
        assert_eq!(
            cma.records[id as usize].age_seconds, 864_000,
            "NONE: the tick still ages the row — strength is what does not move"
        );
        assert_eq!(cma.records[id as usize].tier, TIER_EPISODIC);
    }

    /// (b) EXPONENTIAL with a 100 s half-life. THE LAW IS
    /// `strength *= 0.5^(elapsed_secs / seconds)` — `seconds` is a
    /// HALF-life, so 100 s leaves 0.5 and 200 s leaves 0.25.
    ///
    /// The 0.3679 figure in the brief is 1/e, i.e. the ONE E-FOLD point of
    /// this same curve, reached at ln2 · seconds ≈ 69.3 s of a 100 s
    /// half-life — NOT at 200 s. It is asserted here through the identity
    /// 0.5^(1.4427) = 1/e (1000 s half-life, 1443 s tick) so the
    /// distinction is pinned by arithmetic, not prose.
    #[test]
    fn decay_policy_exponential_is_a_half_life_law() {
        let mut cma = Cma::new(4);
        let id = cma.remember(&unit(0, 4), "half-life fact", "user", 0) as u32;
        assert!(cma.set_decay_params(DECAY_TAG_EXPONENTIAL, 100.0));
        assert_eq!(cma.decay_params(), (DECAY_TAG_EXPONENTIAL, 100.0));

        cma.decay_tick(100);
        assert!(
            (cma.records[id as usize].strength - 0.5).abs() < 1e-6,
            "one 100 s half-life must halve, got {}",
            cma.records[id as usize].strength
        );
        cma.decay_tick(100);
        assert!(
            (cma.records[id as usize].strength - 0.25).abs() < 1e-6,
            "200 s is TWO half-lives: 0.25, got {}",
            cma.records[id as usize].strength
        );

        /* The same 200 s in ONE tick lands in the same place. */
        let mut one = Cma::new(4);
        let oid = one.remember(&unit(0, 4), "half-life fact", "user", 0) as u32;
        assert!(one.set_decay_params(DECAY_TAG_EXPONENTIAL, 100.0));
        one.decay_tick(200);
        assert!(
            (one.records[oid as usize].strength - 0.25).abs() < 1e-6,
            "0.5^(200/100) = 0.25 in a single tick too, got {}",
            one.records[oid as usize].strength
        );

        /* 0.5^(1.4427) = 1/e = 0.367879…: the e-fold lies at ln2
         * half-lives (1443 s of a 1000 s half-life), which is also why
         * "0.3679 after 2 half-lives" is the wrong number. */
        let mut efold = Cma::new(4);
        let eid = efold.remember(&unit(0, 4), "efold", "user", 0) as u32;
        assert!(efold.set_decay_params(DECAY_TAG_EXPONENTIAL, 1000.0));
        efold.decay_tick(1443);
        let s = efold.records[eid as usize].strength;
        assert!(
            (s - 0.367_879_4).abs() < 1e-3,
            "one e-fold is 0.3679 at ln2 half-lives, got {s}"
        );
        let mut two = Cma::new(4);
        let tid = two.remember(&unit(0, 4), "two half-lives", "user", 0) as u32;
        assert!(two.set_decay_params(DECAY_TAG_EXPONENTIAL, 1000.0));
        two.decay_tick(2000);
        let t = two.records[tid as usize].strength;
        assert!((t - 0.25).abs() < 1e-6, "two half-lives is 0.25, got {t}");
        assert!(
            (t - s).abs() > 0.1,
            "the e-fold (0.368) and two half-lives (0.25) are different points on the curve"
        );
    }

    /// (c) TTL: a row STRICTLY older than `seconds` (by `created_at`)
    /// leaves the live set at tick time; a younger one is untouched and
    /// does not decay. `created_at == 0` (unknown — a legacy file) never
    /// expires.
    #[test]
    fn decay_policy_ttl_expires_by_created_at() {
        let mut cma = Cma::new(4);
        let old = cma.remember(&unit(0, 4), "sixty second old", "user", 0) as usize;
        let young = cma.remember(&unit(1, 4), "ten second old", "user", 0) as usize;
        let unknown = cma.remember(&unit(2, 4), "unknown age", "user", 0) as usize;
        let now = now_secs();
        cma.records[old].created_at = now.saturating_sub(60);
        cma.records[young].created_at = now.saturating_sub(10);
        cma.records[unknown].created_at = 0;
        assert!(cma.set_decay_params(DECAY_TAG_TTL, 50.0));

        cma.decay_tick(1);

        assert_eq!(cma.records[old].tier, TIER_FORGOTTEN, "60 s old > 50 s TTL");
        assert_eq!(
            cma.records[old].death_reason.as_deref(),
            Some(TTL_DEATH_REASON),
            "the tick stamps WHY the row died"
        );
        assert_ne!(cma.records[young].tier, TIER_FORGOTTEN, "10 s old < 50 s TTL");
        assert_eq!(cma.records[young].strength, 1.0, "TTL is not a strength decay");
        assert_ne!(
            cma.records[unknown].tier, TIER_FORGOTTEN,
            "created_at 0 = unknown age, never expires"
        );

        /* It left the LIVE set at tick time — recall cannot return it. */
        assert!(
            !cma.recall(&unit(0, 4), 5).iter().any(|h| h.id == old as u32),
            "an expired row must not be recallable"
        );
        /* …and it leaves through the EXISTING accounting: retain() prunes
         * the tombstone and the ledger records why (same path as forget /
         * consolidate). */
        assert_eq!(cma.retain(), 1, "the expired row is the one retain drops");
        assert_eq!(
            cma.deaths.last().map(|d| d.reason.as_str()),
            Some(TTL_DEATH_REASON),
            "the death ledger carries the TTL reason"
        );
        assert_eq!(cma.len(), 2);
        assert_eq!(
            cma.records.iter().filter(|r| r.age_seconds == 1).count(),
            2,
            "the survivors were ticked (age 1) — the TTL pass ran"
        );
    }

    /// (d) an unknown tag is rejected, and so is an impossible parameter,
    /// with the installed policy left UNTOUCHED — a store that re-persists
    /// `decay_params()` after a rejected call must not end up with a
    /// half-installed policy.
    #[test]
    fn decay_policy_rejects_unknown_and_impossible_params() {
        let mut cma = Cma::new(4);
        assert_eq!(
            cma.decay_params(),
            (DECAY_TAG_EXPONENTIAL, 0.0),
            "the shipped default is the historical per-record curve"
        );

        for bogus in [3u8, 7, 42, u8::MAX] {
            assert!(!cma.set_decay_params(bogus, 60.0), "tag {bogus} is not a policy");
        }
        assert_eq!(
            cma.decay_params(),
            (DECAY_TAG_EXPONENTIAL, 0.0),
            "rejections must change nothing"
        );

        /* TTL needs a strictly positive, finite time-to-live. */
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                !cma.set_decay_params(DECAY_TAG_TTL, bad),
                "TTL {bad} must be refused"
            );
        }
        /* EXPONENTIAL's half-life is non-negative and finite. */
        assert!(!cma.set_decay_params(DECAY_TAG_EXPONENTIAL, -5.0));
        assert!(!cma.set_decay_params(DECAY_TAG_EXPONENTIAL, f64::NAN));
        assert_eq!(cma.decay_params(), (DECAY_TAG_EXPONENTIAL, 0.0));

        /* Every accepted pair is read back verbatim (what the store
         * persists, and what a hydrate re-applies). */
        assert!(cma.set_decay_params(DECAY_TAG_TTL, 50.5));
        assert_eq!(cma.decay_params(), (DECAY_TAG_TTL, 50.5));
        assert!(cma.set_decay_params(DECAY_TAG_NONE, 99.0));
        assert_eq!(cma.decay_params(), (DECAY_TAG_NONE, 0.0), "NONE has no seconds");
        assert!(
            cma.set_decay_params(DECAY_TAG_EXPONENTIAL, 0.0),
            "0.0 is the legacy per-record sentinel, not an error"
        );
        assert_eq!(cma.decay_params(), (DECAY_TAG_EXPONENTIAL, 0.0));
    }

    /// (e) THE DEFAULT IS THE OLD BEHAVIOUR: `Cma::new` ships
    /// `(DECAY_TAG_EXPONENTIAL, 0.0)`, where `0.0` means "use each record's
    /// own `half_life`" — byte-for-byte the pre-W5.3 tick
    /// `strength *= exp(-dt / half_life)`, including the zero-half_life
    /// guard and the reinforced/entity curves. Every row is checked
    /// against the closed form, not against another implementation of it.
    #[test]
    fn default_decay_policy_reproduces_the_historical_curve() {
        let mut cma = Cma::new(4);
        let epi = cma.remember(&unit(0, 4), "a turn fact", "user", 0) as u32;
        let ent = cma.remember_entity(&unit(1, 4), "Barnaby is a dog", "Barnaby", "pet") as u32;
        let grown = cma.remember(&unit(2, 4), "a reinforced fact", "user", 0) as u32;
        cma.mark_positive(&[grown]); // doubles the half-life, as recall does
        let zero = cma.remember(&unit(3, 4), "hostile zero half-life", "user", 0) as u32;
        cma.records[zero as usize].half_life = 0;
        cma.records[epi as usize].strength = 0.8; // a partly-decayed row
        assert_eq!(
            cma.decay_params(),
            (DECAY_TAG_EXPONENTIAL, 0.0),
            "the shipped default must BE the historical per-record curve"
        );

        let before: Vec<(u32, f32, u32)> = cma
            .records
            .iter()
            .map(|r| (r.half_life, r.strength, r.age_seconds))
            .collect();
        let dt = 3600u32;
        cma.decay_tick(dt);
        for (i, (hl, s0, age0)) in before.iter().enumerate() {
            let r = &cma.records[i];
            assert_eq!(r.age_seconds, age0 + dt, "row {i}: age advances either way");
            let want = if *hl == 0 {
                *s0 // ml-3 belt: a zero half-life row keeps its strength
            } else {
                s0 * (-(dt as f32) / *hl as f32).exp()
            };
            assert!(
                (r.strength - want).abs() < 1e-6,
                "row {i} (half_life {hl}): default tick must equal exp(-dt/half_life); want {want}, got {}",
                r.strength
            );
        }
        assert!(
            cma.records[ent as usize].strength > 0.999,
            "an entity's 1-year half-life barely moves in an hour"
        );
        assert!(
            cma.records[grown as usize].half_life > 86400,
            "fixture premise: mark_positive grew the half-life"
        );

        /* The persisted pair round-trips: after another policy, restoring
         * the DEFAULT (what re-applying a manifest does) re-arms exactly
         * the same law. */
        let snap: Vec<f32> = cma.records.iter().map(|r| r.strength).collect();
        assert!(cma.set_decay_params(DECAY_TAG_NONE, 0.0));
        cma.decay_tick(86_400);
        assert_eq!(
            cma.records[epi as usize].strength, snap[epi as usize],
            "NONE froze the row"
        );
        assert!(cma.set_decay_params(DECAY_TAG_EXPONENTIAL, 0.0));
        assert_eq!(cma.decay_params(), (DECAY_TAG_EXPONENTIAL, 0.0));
        cma.decay_tick(86_400);
        let hl = cma.records[epi as usize].half_life;
        let want = snap[epi as usize] * (-(86_400f32) / hl as f32).exp();
        assert!(
            (cma.records[epi as usize].strength - want).abs() < 1e-6,
            "restoring the default re-arms the historical law"
        );
    }

    /* ── W5.4: the pin floor is an invariant ─────────────────────────── */

    /// (a) pinning raises a weak row to exactly `PIN_FLOOR` immediately,
    /// and every tick holds it there.
    #[test]
    fn pin_floor_raises_a_weak_row_and_holds_it() {
        let mut cma = Cma::new(4);
        let id = cma.remember(&unit(0, 4), "weak pin candidate", "user", 0) as u32;
        cma.records[id as usize].strength = 0.13;
        assert!(!cma.pin(9_999), "no such record");
        assert!(cma.pin(id), "pinning a live row must succeed");
        assert_eq!(
            cma.records[id as usize].strength, PIN_FLOOR,
            "a pin raises a 0.13 row to the floor ON THE SPOT"
        );
        assert!(is_pinned(&cma.records[id as usize]));

        for i in 0..10 {
            cma.decay_tick(3600);
            assert_eq!(
                cma.records[id as usize].strength, PIN_FLOOR,
                "tick {i}: a pinned row must sit exactly on the floor"
            );
            assert_eq!(cma.records[id as usize].tier, TIER_EPISODIC, "tick {i}");
        }
        assert_eq!(cma.retain(), 0, "nothing to prune: the pin is live at the floor");
        assert_eq!(cma.len(), 1);
    }

    /// (b) a pinned row is never dropped or demoted — not after 100 ticks
    /// of decay, and not under a GLOBAL policy (TTL) that would expire the
    /// whole store.
    #[test]
    fn pinned_row_is_never_dropped_or_expired() {
        let mut cma = Cma::new(4);
        let pin = cma.remember(&unit(0, 4), "pinned anchor", "user", 0) as u32;
        assert!(cma.pin(pin));
        for i in 0..100 {
            cma.decay_tick(86_400);
            assert!(
                cma.records[pin as usize].strength >= PIN_FLOOR - 1e-6,
                "tick {i}: pin strength fell under the floor: {}",
                cma.records[pin as usize].strength
            );
            assert_eq!(
                cma.records[pin as usize].tier, TIER_EPISODIC,
                "tick {i}: a pin is never demoted"
            );
        }
        assert_eq!(cma.retain(), 0, "retain has nothing to drop: the row is pinned");

        /* The global config: TTL 1 s with every row far older than that. */
        cma.records[pin as usize].created_at = now_secs().saturating_sub(3_600);
        assert!(cma.set_decay_params(DECAY_TAG_TTL, 1.0));
        cma.decay_tick(1);
        assert_ne!(
            cma.records[pin as usize].tier, TIER_FORGOTTEN,
            "a TTL broad enough to expire the store must not take a pin"
        );
        assert!(
            cma.records[pin as usize].death_reason.is_none(),
            "no death reason is stamped on a pin"
        );
        assert_eq!(cma.retain(), 0);
        assert_eq!(cma.len(), 1);
        let hits = cma.recall(&unit(0, 4), 3);
        assert_eq!(hits.first().map(|h| h.id), Some(pin), "the pin still recalls");
    }

    /// (c) THE CONTROL: an unpinned row in the same brain does decay away.
    /// Without it, (a)/(b) could pass on a tick that does nothing at all.
    #[test]
    fn unpinned_row_in_the_same_brain_decays_away() {
        let mut cma = Cma::new(4);
        let pin = cma.remember(&unit(0, 4), "pinned anchor", "user", 0) as u32;
        let loose = cma.remember(&unit(1, 4), "idle unpinned twin", "user", 0) as u32;
        assert!(cma.pin(pin));
        for _ in 0..100 {
            cma.decay_tick(86_400);
        }
        assert!(
            cma.records[loose as usize].strength < FORGET_FLOOR,
            "the unpinned twin must decay below the forget floor, got {}",
            cma.records[loose as usize].strength
        );
        assert_eq!(
            cma.records[loose as usize].tier, TIER_EPISODIC,
            "decay does not tombstone — retain does"
        );
        assert_eq!(cma.retain(), 1, "retain drops the decayed twin and nothing else");
        assert_eq!(cma.len(), 1);
        assert!(
            cma.records.iter().any(|r| r.text == "pinned anchor"),
            "the pin outlived the control"
        );
        assert_eq!(cma.deaths.last().map(|d| d.reason.as_str()), Some("decay"));
    }

    /// (d) unpinning restores the ordinary curve: the row decays again from
    /// the floor it was held at, and retain() prunes it once it is under
    /// the forget floor.
    #[test]
    fn unpin_restores_normal_decay() {
        let mut cma = Cma::new(4);
        let id = cma.remember(&unit(0, 4), "released fact", "user", 0) as u32;
        cma.records[id as usize].strength = 0.13;
        assert!(cma.pin(id));
        cma.decay_tick(86_400);
        assert_eq!(cma.records[id as usize].strength, PIN_FLOOR);

        assert!(cma.unpin(id), "unpin must accept a pinned row");
        assert!(
            !is_pinned(&cma.records[id as usize]),
            "every pin spelling was cleared"
        );
        assert!(!cma.unpin(id), "unpinning twice reports 'not pinned'");

        /* One day at the 1-day default: 0.2 · e⁻¹ ≈ 0.0736 — under the
         * floor it was held at, still above the forget floor. */
        cma.decay_tick(86_400);
        let s = cma.records[id as usize].strength;
        assert!(s < PIN_FLOOR, "the released row decays again, got {s}");
        assert!(
            (s - PIN_FLOOR * (-1.0f32).exp()).abs() < 1e-5,
            "the ordinary curve resumes from the floor, got {s}"
        );
        cma.decay_tick(3 * 86_400);
        assert!(cma.records[id as usize].strength < FORGET_FLOOR);
        assert_eq!(cma.retain(), 1, "an unpinned weak row is pruned like any other");
        assert_eq!(cma.len(), 0);

        /* A row pinned by ROLE is unpinned too — the floor must not leak
         * through a marker spelling. */
        let by_role = cma.remember(&unit(2, 4), "role-pinned fact", PIN_ROLE, 0) as u32;
        assert!(is_pinned(&cma.records[by_role as usize]));
        assert!(cma.unpin(by_role));
        assert!(!is_pinned(&cma.records[by_role as usize]));
        assert_eq!(cma.records[by_role as usize].role, "user");
        /* A tombstone is not pinnable. */
        cma.forget(by_role);
        assert!(!cma.pin(by_role), "a forgotten row cannot be pinned");
    }

}

#[cfg(test)]
mod ffi_format_tests {
    use super::*;

    #[test]
    fn records_json_roundtrips_through_hydrate() {
        let mut c = Cma::new(4);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        c.remember(&v, "hello world", "user", 0);
        let json = c.records_json();
        println!("JSON: {}", json);
        let mut c2 = Cma::new(4);
        let mut vecs: Vec<f32> = Vec::new();
        for i in 0..c.len() {
            vecs.extend_from_slice(c.index.vector(i as u32));
        }
        assert!(c2.hydrate(&vecs, c.len(), 4, &json), "hydrate failed");
        assert_eq!(c2.len(), 1);
        assert_eq!(c2.records[0].text, "hello world");
        assert_eq!(c2.records[0].role, "user");
        let hits = c2.recall(&v, 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "hello world");
    }

    /* ── B1.1 / B1.2: read-out + provenance ─────────────────────── */

    /// One-hot vector for the 4-dim test space.
    fn unit(n: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[n % dim] = 1.0;
        v
    }

    #[test]
    fn provenance_is_recorded_and_round_trips() {
        let mut cma = Cma::new(4);
        let v = unit(0, 4);
        let id = cma.remember_with_source(&v, "deploy bucket is sofuu-dist", "user", 0, "chat");
        assert!(id >= 0);
        let rec = &cma.records[id as usize];
        assert_eq!(rec.source, "chat");
        assert!(rec.created_at > 0, "created_at must be stamped at write time");
        assert_eq!(rec.last_recalled_at, 0, "never recalled yet");

        /* recall stamps last_recalled_at */
        let hits = cma.recall(&v, 1);
        assert_eq!(hits.len(), 1);
        assert!(cma.records[id as usize].last_recalled_at > 0);

        /* round trip through the file shape */
        let json = cma.records_json();
        let mut c2 = Cma::new(4);
        assert!(c2.hydrate(&v, 1, 4, &json));
        assert_eq!(c2.records[0].source, "chat");
        assert_eq!(c2.records[0].created_at, cma.records[0].created_at);
        assert_eq!(
            c2.records[0].last_recalled_at,
            cma.records[0].last_recalled_at
        );
    }

    #[test]
    fn legacy_records_hydrate_without_provenance() {
        /* A brain written before B1.2: no created_at / source keys. */
        let meta = r#"{"records":[{"vector_index":0,"kv_page_id":0,"strength":1.0,
            "age_seconds":0,"half_life":86400,"tier":1,"text":"old fact",
            "role":"user","total_recalls":0,"positive_recalls":0}]}"#;
        let mut cma = Cma::new(4);
        assert!(cma.hydrate(&unit(0, 4), 1, 4, meta));
        assert_eq!(cma.records[0].text, "old fact");
        assert_eq!(cma.records[0].created_at, 0, "unknown for legacy records");
        assert_eq!(cma.records[0].source, "");
    }

    #[test]
    fn list_json_filters_and_paginates() {
        /* 16 dims so six one-hot vectors stay distinct — a 4-dim space
         * would collide and the duplicate gate would reinforce instead. */
        let mut cma = Cma::new(16);
        for i in 0..6 {
            let v = unit(i, 16);
            cma.remember(&v, &format!("episode {i}"), "user", 0);
        }
        cma.remember_entity(&unit(7, 16), "Barnaby is a dog", "Barnaby", "pet");

        /* no filter: everything, with the filtered total */
        let all: serde_json::Value =
            serde_json::from_str(&cma.list_json(&ListFilter::default())).unwrap();
        assert_eq!(all["total"], 7);
        assert_eq!(all["records"].as_array().unwrap().len(), 7);

        /* tier filter */
        let entities: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            tier: Some(TIER_ENTITY),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(entities["total"], 1);
        assert_eq!(entities["records"][0]["entity"], "Barnaby");

        /* role filter */
        let users: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            role: Some("user".to_string()),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(users["total"], 6);

        /* pagination: total stays the FILTERED count */
        let page: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            offset: 2,
            limit: 3,
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(page["total"], 7);
        assert_eq!(page["records"].as_array().unwrap().len(), 3);
        assert_eq!(page["records"][0]["text"], "episode 2");

        /* text filter (case-insensitive substring) */
        let hits: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            text: Some("EPISODE 3".to_string()),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(hits["total"], 1);
        assert_eq!(hits["records"][0]["text"], "episode 3");

        /* forgotten tombstones are hidden by default, visible on request */
        let victim = all["records"][0]["id"].as_u64().unwrap() as u32;
        assert!(cma.forget(victim));
        let after: serde_json::Value =
            serde_json::from_str(&cma.list_json(&ListFilter::default())).unwrap();
        assert_eq!(after["total"], 6, "the tombstone must leave the default view");
        let tombstones: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            tier: Some(TIER_FORGOTTEN),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(tombstones["total"], 1, "requesting the tier shows it");

        /* min_strength filter */
        cma.decay_tick(86400 * 3);
        let strong: serde_json::Value = serde_json::from_str(&cma.list_json(&ListFilter {
            min_strength: Some(0.5),
            ..Default::default()
        }))
        .unwrap();
        assert!(strong["total"].as_u64().unwrap() < 7, "decay must thin the list");
    }

    #[test]
    fn export_view_pairs_records_with_vectors() {
        let mut cma = Cma::new(16);
        for i in 0..4 {
            cma.remember(&unit(i, 16), &format!("m{i}"), "user", 0);
        }
        assert!(cma.forget(1));
        let filter = ListFilter::default();
        let view: serde_json::Value = serde_json::from_str(&cma.list_json(&filter)).unwrap();
        let n = view["records"].as_array().unwrap().len();
        let flat = cma.flat_vectors_for(&filter);
        assert_eq!(n, 3, "tombstone hidden from the view");
        assert_eq!(
            flat.len(),
            n * 16,
            "vectors must pair 1:1 with the view's records"
        );
        /* The unfiltered table stays aligned with the FILE (4 records). */
        assert_eq!(cma.flat_vectors().len(), 4 * 16);
    }

    #[test]
    fn import_merges_duplicates_and_keeps_provenance() {
        let mut dst = Cma::new(16);
        dst.remember(&unit(0, 16), "existing fact", "user", 0);

        /* A card: one duplicate of the existing fact (stronger, older) and
         * one new record with provenance. */
        let card = serde_json::json!({
            "records": [
                {"text": "existing fact", "tier": 1, "strength": 0.9,
                 "created_at": 1000, "role": "user", "source": "card"},
                {"text": "imported fact", "tier": 2, "strength": 0.5,
                 "created_at": 2000, "role": "system", "source": "card"}
            ]
        })
        .to_string();
        let mut vecs = unit(0, 16);
        vecs.extend_from_slice(&unit(5, 16));
        let stats = dst.import_records(&vecs, 2, 16, &card);
        assert_eq!(stats.added, 1, "one new record");
        assert_eq!(stats.merged, 1, "the duplicate merged");
        assert_eq!(dst.len(), 2);

        /* The merged record kept its identity but took the earlier date. */
        let merged = &dst.records[0];
        assert_eq!(merged.text, "existing fact");
        assert_eq!(merged.created_at, 1000, "earliest sighting wins");
        assert_eq!(merged.source, "", "original provenance kept");

        /* The new record carried the card's tier + provenance. */
        let imported = &dst.records[1];
        assert_eq!(imported.tier, TIER_SEMANTIC);
        assert_eq!(imported.source, "card");
        assert_eq!(imported.created_at, 2000);
        assert!((imported.strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn import_skips_tombstones_and_dim_mismatch() {
        let mut cma = Cma::new(16);
        let card = serde_json::json!({
            "records": [
                {"text": "dead", "tier": 3},
                {"text": "live", "tier": 1}
            ]
        })
        .to_string();
        let vecs = [unit(1, 16), unit(2, 16)].concat();
        let stats = cma.import_records(&vecs, 2, 16, &card);
        assert_eq!(stats.skipped, 1, "tombstones never import");
        assert_eq!(stats.added, 1);

        /* A card in another space is refused wholesale, not half-merged. */
        let before = cma.len();
        let bad = cma.import_records(&vecs, 2, 8, &card);
        assert_eq!(bad, ImportStats::default());
        assert_eq!(cma.len(), before);
    }

    #[test]
    fn flat_vectors_matches_records() {
        let mut cma = Cma::new(4);
        cma.remember(&[1.0, 0.0, 0.0, 0.0], "a", "user", 0);
        cma.remember(&[0.0, 1.0, 0.0, 0.0], "b", "user", 0);
        let flat = cma.flat_vectors();
        assert_eq!(flat, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn hydrate_entity_and_corecall_fields() {
        let mut c = Cma::new(4);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        c.remember_entity(&v, "Barnaby is a dog", "Barnaby", "pet");
        // recall once to populate corecall + counters
        c.recall(&v, 3);
        c.mark_positive(&[0]);
        let json = c.records_json();
        let mut c2 = Cma::new(4);
        let mut vecs: Vec<f32> = Vec::new();
        for i in 0..c.len() {
            vecs.extend_from_slice(c.index.vector(i as u32));
        }
        assert!(c2.hydrate(&vecs, c.len(), 4, &json), "hydrate failed: {}", json);
        assert_eq!(c2.records[0].tier, TIER_ENTITY);
        assert_eq!(c2.records[0].entity_name.as_deref(), Some("Barnaby"));
        assert_eq!(c2.records[0].positive_recalls, 1);
        assert_eq!(c2.records[0].total_recalls, 1);
    }
}
