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
        }
    }
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

pub struct Cma {
    pub vec_dim: usize,
    pub index: Hnsw,
    pub records: Vec<Record>,
    /// Vector store parallel to records (index by record id).
    pub vectors: Vec<Vec<f32>>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    corecall_count: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    corecall_rpos: Option<u8>,
    total_recalls: u16,
    positive_recalls: u16,
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
            vectors: Vec::new(),
        }
    }

    /// Hydrate from a loaded brain file: the flat f32 vector table + the
    /// metadata JSON (C's `cma_qtsq_load` output). Returns None on a
    /// malformed payload (caller falls back to a fresh store).
    pub fn hydrate(&mut self, vecs: &[f32], n: usize, dim: usize, meta_json: &str) -> bool {
        if dim != self.vec_dim || vecs.len() != n * dim {
            return false;
        }
        // The brain-file metadata is {"records":[...]} — the same shape the
        // C cma_flush wrote.
        let records: Vec<RecordJson> = match serde_json::from_str::<serde_json::Value>(meta_json) {
            Ok(v) => match v.get("records").and_then(|r| serde_json::from_value(r.clone()).ok()) {
                Some(r) => r,
                None => return false,
            },
            Err(_) => return false,
        };
        for i in 0..n {
            let v: Vec<f32> = vecs[i * dim..(i + 1) * dim].to_vec();
            self.vectors.push(v.clone());
            self.index.add_vector(&v);
            let mut rec = Record::new(i as u32, "", "", 0);
            if let Some(rj) = records.get(i) {
                rec.kv_page_id = rj.kv_page_id;
                rec.strength = rj.strength;
                rec.age_seconds = rj.age_seconds;
                rec.half_life = rj.half_life;
                rec.tier = rj.tier;
                rec.text = rj.text.clone();
                rec.role = rj.role.clone();
                rec.entity_name = rj.entity_name.clone();
                rec.entity_type = rj.entity_type.clone();
                rec.corecall_ids = rj.corecall_ids.clone();
                rec.total_recalls = rj.total_recalls;
                rec.positive_recalls = rj.positive_recalls;
            } else {
                rec.text = "(corrupt record)".to_string();
            }
            self.records.push(rec);
        }
        true
    }

    /// Serialize records to the brain-file metadata JSON (matches C).
    pub fn records_json(&self) -> String {
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
                corecall_count: if r.corecall_ids.is_empty() { None } else { Some(r.corecall_ids.len() as u8) },
                corecall_rpos: if r.corecall_ids.is_empty() { None } else { Some((r.corecall_ids.len() % 8) as u8) },
                total_recalls: r.total_recalls,
                positive_recalls: r.positive_recalls,
            })
            .collect();
        serde_json::json!({ "records": arr }).to_string()
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
            let rec = &mut self.records[id as usize];
            if rec.tier == TIER_FORGOTTEN {
                continue;
            }
            if rec.kv_page_id > 0 && seen.insert(rec.kv_page_id) {
                rec.strength = 1.0;
                rec.half_life = (rec.half_life as f32 * 1.2) as u32;
                out.push(rec.kv_page_id);
                if out.len() >= n {
                    break;
                }
            }
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
                        return *dup_id as i32;
                    }
                }
            }
        }
        let id = self.records.len() as u32;
        let rec = Record::new(id, text, role, kv_page_id);
        self.records.push(rec);
        self.vectors.push(vec.to_vec());
        self.index.add_vector(vec);
        id as i32
    }

    /// Recall top-k memories with composite scoring (strength × tier ×
    /// resonance), gravity boost, and reinforcement.
    pub fn recall(&mut self, query: &[f32], top_k: usize) -> Vec<RecallHit> {
        if self.records.is_empty() {
            return Vec::new();
        }
        let ef = (top_k * 3).max(32);
        let mut hits = self.index.search(query, ef);
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
            scored.push((id, dist, score));
        }
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Gravity: co-recall partners of the top-1 get a boost.
        if scored.len() > 1 {
            let top1 = scored[0].0;
            let partners: Vec<u32> = self.records[top1 as usize].corecall_ids.clone();
            for (id, _, score) in scored.iter_mut().skip(1) {
                if partners.contains(id) {
                    *score *= 1.0 + GRAVITY_G;
                }
            }
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        }

        let n = scored.len().min(top_k);
        let mut out = Vec::with_capacity(n);
        let mut recalled_ids = Vec::with_capacity(n);
        for (id, dist, score) in scored.into_iter().take(n) {
            // Reinforce + track recall.
            let rec = &mut self.records[id as usize];
            rec.strength = (rec.strength * 1.1).min(1.0);
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
        }
        self.update_gravity(&recalled_ids);
        out
    }

    /// Ebbinghaus decay: strength *= exp(-dt / half_life).
    pub fn decay_tick(&mut self, dt_seconds: u32) {
        for rec in self.records.iter_mut() {
            if rec.tier == TIER_FORGOTTEN {
                continue;
            }
            rec.age_seconds = rec.age_seconds.saturating_add(dt_seconds);
            let f = (-(dt_seconds as f32) / rec.half_life as f32).exp();
            rec.strength *= f;
        }
    }

    /// Mark memories as positively reinforced (user continued).
    pub fn mark_positive(&mut self, ids: &[u32]) -> u32 {
        let mut marked = 0;
        for &id in ids {
            if let Some(rec) = self.records.get_mut(id as usize) {
                if rec.positive_recalls < 65535 {
                    rec.positive_recalls += 1;
                    marked += 1;
                }
            }
        }
        marked
    }

    pub fn forget(&mut self, id: u32) -> bool {
        if let Some(rec) = self.records.get_mut(id as usize) {
            rec.tier = TIER_FORGOTTEN;
            true
        } else {
            false
        }
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
        for rec in self.records.drain(..) {
            if is_dead(&rec) {
                continue;
            }
            remap.insert(rec.id, kept_records.len() as u32);
            kept_vectors.push(self.vectors[rec.id as usize].clone());
            kept_records.push(rec);
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
        self.vectors = kept_vectors;
        // Rebuild the HNSW store over survivors (ids are positions again).
        self.index = Hnsw::new(self.vec_dim);
        for v in &self.vectors {
            self.index.add_vector(v);
        }
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
                let id = rec.id as usize;
                self.vectors[id] = vec.to_vec();
                // Re-index: rebuild the HNSW store is heavy; for entity updates
                // we keep the old vector (acceptable drift, matches C which
                // also just memcpys into the store without re-adding).
                return rec.id as i32;
            }
        }
        let idx = self.remember(vec, text, "entity", 0);
        if idx >= 0 {
            let rec = &mut self.records[idx as usize];
            rec.tier = TIER_ENTITY;
            rec.half_life = 86400 * 365; // entities never decay
            rec.entity_name = Some(name.to_string());
            rec.entity_type = Some(etype.to_string());
        }
        idx
    }

    /// Consolidate weak episodic memories into semantic clusters (k-means +
    /// dream summarization).
    pub fn consolidate(&mut self) -> usize {
        let n = self.records.len();
        if n == 0 {
            return 0;
        }
        // Find weak episodic memories (strength < 0.25, age > 1 day).
        let weak: Vec<u32> = self
            .records
            .iter()
            .filter(|r| r.tier == TIER_EPISODIC && r.strength < 0.25 && r.age_seconds > 86400)
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
                self.vectors[weak[(rng as usize) % weak.len()] as usize].clone()
            })
            .collect();
        let mut counts = vec![0usize; k];
        let mut assignments = vec![0usize; weak.len()];

        for _ in 0..10 {
            let mut sums: Vec<Vec<f32>> = vec![vec![0.0; self.vec_dim]; k];
            counts = vec![0usize; k];
            for (i, &wid) in weak.iter().enumerate() {
                let v = &self.vectors[wid as usize];
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

        // Re-assign final + build clusters; dream-summarize each.
        let mut consolidated = 0usize;
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
            let new_idx = self.remember(&centroids[ci], &summary, "system", 0);
            if new_idx >= 0 {
                let rec = &mut self.records[new_idx as usize];
                rec.tier = TIER_SEMANTIC;
                rec.half_life = 86400 * 30;
                consolidated += 1;
            }
        }

        // Mark originals forgotten.
        for &wid in &weak {
            self.records[wid as usize].tier = TIER_FORGOTTEN;
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
        assert_eq!(cma.vectors.len(), 3);
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
        let vecs: Vec<f32> = c.vectors.iter().flatten().copied().collect();
        assert!(c2.hydrate(&vecs, c.vectors.len(), 4, &json), "hydrate failed");
        assert_eq!(c2.len(), 1);
        assert_eq!(c2.records[0].text, "hello world");
        assert_eq!(c2.records[0].role, "user");
        let hits = c2.recall(&v, 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "hello world");
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
        let vecs: Vec<f32> = c.vectors.iter().flatten().copied().collect();
        assert!(c2.hydrate(&vecs, c.vectors.len(), 4, &json), "hydrate failed: {}", json);
        assert_eq!(c2.records[0].tier, TIER_ENTITY);
        assert_eq!(c2.records[0].entity_name.as_deref(), Some("Barnaby"));
        assert_eq!(c2.records[0].positive_recalls, 1);
        assert_eq!(c2.records[0].total_recalls, 1);
    }
}
