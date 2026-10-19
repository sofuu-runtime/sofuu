// sofuu-core — CMA (Cognitive Memory Architecture).
//
// Rust port of src/memory/mod_memory.c — the 4-tier memory model:
// working → episodic → semantic → entity, with Ebbinghaus decay,
// k-means consolidation, TF-IDF "dream" summarization, co-recall gravity,
// and resonance scoring. Pure logic (no QuickJS) — the JS bindings stay in
// C (mod_memory.c) via FFI, or can be re-exposed later.

use crate::memory::hnsw::Hnsw;
use std::collections::HashMap;

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
pub const TOKEN_RERANK_BOOST: f32 = 0.35;

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
    pub score: f32,
    pub role: String,
    pub text: String,
    pub tier: u8,
    pub strength: f32,
    /// Set for TIER_ENTITY records — the entity namespace the record
    /// belongs to (e.g. "agent:researcher"). Lets JS-side consumers filter
    /// per-agent scopes out of one physical brain (PLAN-AGENTS A3).
    pub entity: Option<String>,
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

pub struct Cma {
    pub vec_dim: usize,
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
}

/// Above this many dirty ids, tracking per-record stops paying — the next
/// flush compacts instead (one pass over live records).
const DIRTY_TRACK_CAP: usize = 4096;

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
            index: Hnsw::new(vec_dim),
            records: Vec::new(),
            dirty_ids: Vec::new(),
            layout_dirty: false,
            ef_search: None,
            diversity_floor: Some(DIVERSITY_FLOOR_DEFAULT),
            working: Vec::new(),
            working_next_seq: 0,
            deaths: Vec::new(),
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
    pub fn remember_working(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
        source: &str,
    ) -> u32 {
        debug_assert_eq!(vec.len(), self.vec_dim);
        /* Working-set near-dup gate (same DUP_DISTANCE as the persistent
         * one): the newer sighting refreshes the timestamp so the fact
         * stays fresh, and the caller gets the SAME id back. */
        for w in &mut self.working {
            let d = sq_dist(w.vector.as_slice(), vec).sqrt();
            if d < DUP_DISTANCE {
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
        if dim != self.vec_dim || vecs.len() != n * dim {
            return false;
        }
        // The brain-file metadata is {"records":[...]} — the same shape the
        // C cma_flush wrote (M2: plus an optional top-level "deaths").
        let meta: serde_json::Value = match serde_json::from_str(meta_json) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let records: Vec<RecordJson> =
            match meta.get("records").and_then(|r| serde_json::from_value(r.clone()).ok()) {
                Some(r) => r,
                None => return false,
            };
        for i in 0..n {
            self.index.add_vector(&vecs[i * dim..(i + 1) * dim]);
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
        if let Some(d) = meta.get("deaths").and_then(|x| serde_json::from_value::<Vec<DeathEntry>>(x.clone()).ok()) {
            self.deaths = d;
            if self.deaths.len() > DEATH_LEDGER_CAP {
                self.deaths = self.deaths.split_off(self.deaths.len() - DEATH_LEDGER_CAP);
            }
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

            /* Dedup: a near-duplicate of an existing live record merges. */
            let dup = if self.records.is_empty() {
                None
            } else {
                self.index
                    .search(v, 1)
                    .first()
                    .copied()
                    .filter(|(id, d)| {
                        *d < DUP_DISTANCE
                            && self.records[*id as usize].tier != TIER_FORGOTTEN
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
    pub fn remember(&mut self, vec: &[f32], text: &str, role: &str, kv_page_id: u32) -> i32 {
        self.remember_with_source(vec, text, role, kv_page_id, "")
    }

    /// B1.2: remember + record where the memory came from. A near-duplicate
    /// keeps its ORIGINAL source (the first sighting is the provenance).
    pub fn remember_with_source(
        &mut self,
        vec: &[f32],
        text: &str,
        role: &str,
        kv_page_id: u32,
        source: &str,
    ) -> i32 {
        debug_assert_eq!(vec.len(), self.vec_dim);
        // Near-duplicate gate: reinforce if a very close memory exists.
        if !self.records.is_empty() {
            let hits = self.index.search(vec, 1);
            if let Some((dup_id, dist)) = hits.first() {
                if *dist < DUP_DISTANCE {
                    let rec = &mut self.records[*dup_id as usize];
                    if rec.tier != TIER_FORGOTTEN {
                        rec.strength = 1.0;
                        rec.age_seconds = 0;
                        let id = *dup_id;
                        self.mark_dirty(id);
                        return id as i32;
                    }
                }
            }
        }
        let id = self.records.len() as u32;
        let mut rec = Record::new(id, text, role, kv_page_id);
        rec.source = source.to_string();
        self.records.push(rec);
        self.index.add_vector(vec);
        self.mark_dirty(id);
        id as i32
    }

    /// Recall top-k memories with composite scoring (strength × tier ×
    /// resonance), gravity boost, and reinforcement.
    pub fn recall(&mut self, query: &[f32], top_k: usize) -> Vec<RecallHit> {
        if self.records.is_empty() && self.working.is_empty() {
            return Vec::new();
        }
        /* Candidate pool: the search returns the closest `pool` records.
         * B5.3: SOFUU_BRAIN_EF_SEARCH (or set_ef_search) widens the search
         * itself; unset keeps the pre-B5 width exactly. */
        let pool = (top_k * 3).max(32);
        let ef = self.ef_search.unwrap_or((pool * 4).max(8));
        let mut hits = self.index.search_with_ef(query, pool, ef);
        let now = now_secs();
        // Filter forgotten + too-weak, score.
        let mut scored: Vec<(u32, f32, f32)> = Vec::new(); // (id, dist, score)
        for (id, dist) in hits.drain(..) {
            let rec = &self.records[id as usize];
            if rec.tier == TIER_FORGOTTEN || rec.strength < FORGET_FLOOR {
                continue;
            }
            let tier_w = match rec.tier {
                TIER_WORKING => 0.5,
                TIER_SEMANTIC => 1.0,
                TIER_ENTITY => 1.5,
                _ => 0.8,
            };
            let norm = dist / (dist + 1.0);
            let mut score = (1.0 - norm) * rec.strength * tier_w;
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

    /// M1.3 (Brain Core II): query-token re-rank — the recall for callers
    /// that also HAVE the query text. Vector similarity is the retrieval
    /// engine, but a chatty generic record can outrank the one record
    /// carrying the query's distinctive token ("quokka"): the embedder
    /// smooths over exactly the tokens that make a hit THE answer. The law
    /// is small and provably near-inert:
    ///
    ///   mult = 1 + TOKEN_RERANK_BOOST × Σ rarity(t)  for query tokens t
    ///   rarity(t) = 1 − (hits containing t) / (all hits)
    ///
    /// rarity is computed over the CANDIDATE POOL, so boilerplate ("the",
    /// "about") weighs nothing wherever it appears and a token unique to
    /// one hit weighs full. The re-rank never jumps the reflex partition
    /// (an exact-distance match stays above every inexact one) and is
    /// exactly inert for a query with no rare tokens — same order in,
    /// same order out.
    pub fn recall_tokens(
        &mut self,
        query: &[f32],
        top_k: usize,
        query_text: &str,
    ) -> Vec<RecallHit> {
        /* ≥3 chars: two-letter words ride idf≈0 anyway, but dropping them
         * here keeps the ×(1−df/n) work off the noise floor. */
        let q_tokens: Vec<String> = tokenize(query_text)
            .into_iter()
            .filter(|t| t.chars().count() >= 3)
            .collect();
        if q_tokens.is_empty() {
            return self.recall(query, top_k);
        }
        let candidate_k = if top_k < 5 { (top_k * 3).max(6) } else { top_k };
        let mut hits = self.recall(query, candidate_k);
        if hits.len() < 2 {
            hits.truncate(top_k);
            return hits;
        }
        let n = hits.len();
        let mut rarity: Vec<f32> = Vec::with_capacity(q_tokens.len());
        for t in &q_tokens {
            let df = hits
                .iter()
                .filter(|h| tokenize(&h.text).iter().any(|ht| ht == t))
                .count();
            rarity.push(1.0 - (df as f32 / n as f32));
        }
        const REFLEX_EPS_TOKEN: f32 = 0.02;
        let mut exact: Vec<(RecallHit, f32)> = Vec::new();
        let mut inexact: Vec<(RecallHit, f32)> = Vec::new();
        for h in hits.drain(..) {
            let ht = tokenize(&h.text);
            let m = 1.0
                + TOKEN_RERANK_BOOST
                    * q_tokens
                        .iter()
                        .zip(rarity.iter())
                        .filter(|(t, _)| ht.iter().any(|x| x == *t))
                        .map(|(_, r)| *r)
                        .sum::<f32>();
            let boosted = h.score * m.min(1.0 + TOKEN_RERANK_BOOST);
            if h.distance <= REFLEX_EPS_TOKEN {
                exact.push((h, boosted));
            } else {
                inexact.push((h, boosted));
            }
        }
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        inexact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::with_capacity(top_k.min(n));
        for (mut h, s) in exact.into_iter().chain(inexact.into_iter()).take(top_k) {
            h.score = s;
            out.push(h);
        }
        out
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

    /// Ebbinghaus decay: strength *= exp(-dt / half_life).
    pub fn decay_tick(&mut self, dt_seconds: u32) {
        for rec in self.records.iter_mut() {
            if rec.tier == TIER_FORGOTTEN {
                continue;
            }
            rec.age_seconds = rec.age_seconds.saturating_add(dt_seconds);
            // ml-3 (AUDIT-2026-09-07) belt: half_life 0 would make the factor
            // exp(-inf) = 0 — an instant permanent erase. hydrate normalizes
            // the file value; decay stays multiplicative-neutral, not fatal.
            if rec.half_life == 0 {
                continue;
            }
            let f = (-(dt_seconds as f32) / rec.half_life as f32).exp();
            rec.strength *= f;
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
            self.mark_dirty(id);
            true
        } else {
            false
        }
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
            _ => rec.strength < FORGET_FLOOR && rec.role != PIN_ROLE,
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
                dead_notes.push((rec.text.clone(), reason));
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
        for rec in self.records.iter_mut() {
            if rec.tier == TIER_ENTITY
                && rec.entity_name.as_deref() == Some(name)
            {
                rec.text = text.to_string();
                rec.strength = 1.0;
                rec.age_seconds = 0;
                // ml-5 (AUDIT-2026-09-07): the old upsert replaced only the
                // side Vec and left the search store stale — the drift the
                // audit flagged. The HNSW store IS the single vector store;
                // set_vector lands the upsert there (search sees the new
                // coordinates; neighbor links keep old-geometry decisions,
                // the same trade C makes by memcpying without re-adding).
                self.index.set_vector(rec.id, vec);
                let id = rec.id;
                self.mark_dirty(id);
                return id as i32;
            }
        }
        /* Entity identity lands ONLY on a genuinely fresh insert. If the
         * dup gate merged us into an existing row, re-tiering it here
         * would kidnap that row into the entity class and rename it on
         * the next upsert of the same name — the M0 fuzz caught a pin
         * (vectors accidentally identical) converted this way, then its
         * text replaced. A merged result is a reinforcement of the
         * survivor; its identity is not ours to rewrite. */
        let before = self.records.len();
        let idx = self.remember(vec, text, "entity", 0);
        if idx >= 0 && (idx as usize) == before && self.records.len() == before + 1 {
            let rec = &mut self.records[idx as usize];
            rec.tier = TIER_ENTITY;
            rec.half_life = 86400 * 365; // entities never decay
            rec.entity_name = Some(name.to_string());
            rec.entity_type = Some(etype.to_string());
        }
        idx
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
        let id = self.remember(vec, text, "agent_turn", 0);
        if id >= 0 && (id as usize) == before && self.records.len() == before + 1 {
            let rec = &mut self.records[id as usize];
            rec.entity_name = Some(name.to_string());
            rec.entity_type = Some(etype.to_string());
            rec.tier = TIER_EPISODIC; // decays on the 1-day curve, capacity-evictable
        }
        id
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
                    && r.role != PIN_ROLE
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
            let rec = &mut self.records[wid as usize];
            rec.tier = TIER_FORGOTTEN;
            /* M2: a consolidated fact is not lost — it lives inside the
             * semantic summary — so the reason says exactly that. */
            rec.death_reason = Some("consolidated".to_string());
            self.mark_dirty(wid);
        }
        let mut consolidated = 0usize;
        for (ci, summary) in summaries {
            let before = self.records.len();
            let new_idx = self.remember(&centroids[ci], &summary, "system", 0);
            if new_idx < 0 {
                continue;
            }
            /* Re-tier only a genuinely FRESH insert: a merge into an
             * existing (alive) record reinforced it — re-tiering that
             * record here would overwrite whatever it was. */
            if self.records.len() > before && (new_idx as usize) == before {
                let rec = &mut self.records[new_idx as usize];
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

    #[test]
    fn duplicate_reinforces() {
        let mut cma = Cma::new(4);
        let mut v = vec![0.0; 4];
        v[0] = 1.0;
        cma.remember(&v, "a", "user", 0);
        cma.remember(&v, "a again", "user", 0);
        // Near-duplicate → same id, not a new record.
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
        /* The working near-dup gate returns the SAME id (one fact). */
        let again = cma.remember_working(&v, "session fact again", "user", 0, "chat");
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
        // An entity whose vector EXACTLY equals the pin's: the dup gate
        // merges, and the pin row must come out untouched.
        let got = cma.remember_entity(&pv, "entity kidnap fact", "ent-X", "fuzz");
        assert_eq!(got as usize, pin, "identical vector must merge, not insert");
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
