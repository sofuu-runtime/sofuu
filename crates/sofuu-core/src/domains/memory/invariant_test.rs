// sofuu-core — Brain Core II M0: the CMA invariant fuzz suite.
//
// Model-based property testing: a deterministic xorshift drives 12k
// adversarial operations (remember / duplicate-remember / recall /
// mark_positive / forget / decay / retain / consolidate / entity upsert /
// working churn + promotion / flush→reload) against a real Cma, and the
// suite asserts the invariants every consumer of the brain depends on.
// Its whole job is to break changes before users do: a milestone that
// violates any invariant here does not ship.
//
// Run: cargo test -p sofuu-core --release -- fuzz_cma
//
// The invariants (the product contract):
//   I1  ids are dense — record.position == record.id, always
//   I2  a forgotten text never resurfaces through any INDIRECT path
//       (retain, consolidation, capacity behavior, flush+reload). It may
//       return only via an explicit remember() of that same text — a
//       /remember after /forget is a legal new fact.
//   I3  a live record above the forget floor is among the top-8 hits for
//       its OWN vector, carrying its exact text
//   I4  flush → hydrate round-trips count, strengths, texts, provenance
//   I5  the working set never exceeds WORKING_MAX_RECORDS
//   I6  promotion preserves the working record's original created_at and
//       source and lands episodic
//   I7  mark_positive changes counters, never strength
//   I8  pins and entities that were NOT explicitly forgotten survive
//       retain at any strength (forgetting one is legal — the fuzz just
//       never does, so the survival claim stays decidable)
//   I9  the default list view never shows a forgotten-tier record

use super::cma::{is_pinned, Cma, ListFilter, FORGET_FLOOR, PIN_ROLE, TIER_FORGOTTEN, WORKING_MAX_RECORDS};
use std::collections::{HashMap, HashSet};

const DIM: usize = 64;
const OPS: usize = 12_000;
const TEXT_POOL: usize = 3_000;
const ENTITY_POOL: usize = 40;
const FLUSH_PERIOD: usize = 250;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn usize(&mut self, n: usize) -> usize {
        (self.next() % (n as u32)) as usize
    }
    fn pick(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        let x = (self.next() as f64 / u32::MAX as f64) * total;
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if x < acc {
                return i;
            }
        }
        probs.len() - 1
    }
}

/// Injective AND quasi-orthogonal for keys < DIM²: a UNIQUE pair of basis
/// spikes per key. The lo spike carries 1.0 and the hi spike 0.5, so the
/// pair is ORDERED — key k1 (lo 16, hi 25) and key k2 (lo 25, hi 16) stay
/// distinct (an earlier all-1.0 encoding collided them at cosine 1.0, and
/// remember_entity's dup gate then kidnapped the row). Distinct keys share
/// at most one spike, cosine ≤ 0.8, Euclidean distance ≥ ~0.63 — far above
/// the 0.05 dup gate — while re-remembering the SAME key is an exact match
/// the gate is supposed to catch.
fn vec_for(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i % DIM] = 1.0;
    v[DIM - 1 - (i / DIM) % (DIM - 1)] = 0.5;
    v
}

fn text_for(i: usize) -> String {
    format!("fact-{i}-{}", "x".repeat(i % 7))
}

#[test]
fn fuzz_cma_product_invariants() {
    for seed in [0x9e37_79b9u32, 0x1234_5678, 0xdead_beef] {
        run_fuzz(seed);
    }
}

fn run_fuzz(seed: u32) {
    let mut rng = Rng(seed);

    let mut cma = Cma::new(DIM);
    // Shadow state
    let mut dead: HashSet<String> = HashSet::new();
    let mut pins: HashSet<String> = HashSet::new(); // pin texts (renumber-proof)
    let mut entities: HashMap<String, u32> = HashMap::new(); // name -> id
    let mut working_ids: Vec<u32> = Vec::new();
    let mut next_text = 0usize;

    // Live texts snapshot, rebuilt lazily (a full O(n) set per op at 4k
    // rows × 12k ops is spendy; only the prune paths need the diff).
    let live_texts = |cma: &Cma| -> HashSet<String> {
        cma.records
            .iter()
            .filter(|r| r.tier != TIER_FORGOTTEN)
            .map(|r| r.text.clone())
            .collect()
    };

    for op in 0..OPS {
        let choice = rng.pick(&[
            0.28, // remember (fresh or legal resurrection)
            0.10, // re-remember a stored text (dup-gate / reinforcement)
            0.22, // recall (I2 + I3)
            0.10, // mark_positive (I7)
            0.09, // forget (I2 activation; never pins/entities)
            0.07, // decay
            0.04, // retain (I2 capture + I8)
            0.02, // consolidate (rare)
            0.04, // entity upsert (I8 subjects)
            0.04, // working churn + promotion (I5/I6)
        ]);
        let snapshot_live: HashSet<String> = live_texts(&cma);

        match choice {
            // ── remember (fresh or legal resurrection) ──────────────
            0 => {
                let i = if rng.usize(4) != 0 || next_text >= TEXT_POOL {
                    let i = next_text % TEXT_POOL;
                    next_text += 1;
                    i
                } else {
                    // explicit re-remember of a possibly-dead text
                    let i = rng.usize(TEXT_POOL);
                    dead.remove(&text_for(i));
                    i
                };
                let t = text_for(i);
                let pin = rng.usize(8) == 0;
                let before = cma.len();
                let id = if pin {
                    cma.remember_with_source(&vec_for(i), &t, PIN_ROLE, 0, "fuzz-pin")
                } else {
                    cma.remember(&vec_for(i), &t, "user", 0)
                };
                assert!(id >= 0, "op {op}: remember failed");
                let id = id as usize;
                if cma.len() == before + 1 {
                    // Appended: a genuinely new row (fresh text, or an
                    // explicit re-remember of a forgotten text arriving as
                    // a new record).
                    assert_eq!(id, before, "op {op}: appended row not at the tail");
                    assert_eq!(cma.records[id].text, t, "op {op}: fresh row altered text");
                    // An explicit remember() is always a legal (re-)birth
                    // of that text.
                    dead.remove(&t);
                    if pin {
                        pins.insert(t);
                    }
                } else {
                    // Reinforced an existing LIVE row in place — the dup
                    // lane keeps the original text/role, so records[id]
                    // may legitimately predate this call (even when id
                    // happens to be the tail row).
                    assert_ne!(
                        cma.records[id].tier,
                        TIER_FORGOTTEN,
                        "op {op}: dup merge landed on a tombstone"
                    );
                    dead.remove(&t);
                }
            }
            // ── re-remember an already-stored text (merge path) ─────
            1 => {
                if !cma.records.is_empty() {
                    let id = rng.usize(cma.records.len());
                    if cma.records[id].tier != TIER_FORGOTTEN {
                        let t = cma.records[id].text.clone();
                        let v = cma.index.vector(id as u32).to_vec();
                        let before = cma.len();
                        let got = cma.remember(&v, &t, "user", 0) as usize;
                        if cma.len() == before {
                            assert_ne!(
                                cma.records[got].tier,
                                TIER_FORGOTTEN,
                                "op {op}: merge hit a tombstone"
                            );
                            assert_eq!(
                                texts_equal(&cma, got, &t),
                                true,
                                "op {op}: merge returned a record with different text"
                            );
                        }
                    }
                }
            }
            // ── recall (I2 + I3) ────────────────────────────────────
            2 => {
                let hits = cma.recall(&vec_for(rng.usize(TEXT_POOL + 8)), 8);
                for h in &hits {
                    if dead.contains(&h.text) {
                        panic!(
                            "op {op} I2 RESURRECTION: '{}' recalled (id {}, tier {})",
                            h.text, h.id, h.tier
                        );
                    }
                }
                // I3: a sampled surviving record finds itself in top-8
                if !cma.records.is_empty() {
                    let id = rng.usize(cma.records.len());
                    let (rtext, rstrength, rtier) = {
                        let r = &cma.records[id];
                        (r.text.clone(), r.strength, r.tier)
                    };
                if rtier != TIER_FORGOTTEN && rstrength >= FORGET_FLOOR && !is_pinned(&cma.records[id]) {
                    let v = cma.index.vector(id as u32).to_vec();
                    let hits = cma.recall(&v, 8);
                    /* The contract is about the FACT, not the row id: an
                     * exact clone (op-1 re-remember after a dup-gate miss)
                     * is suppressed by the diversity floor in favor of the
                     * canonical sibling — the same text must come back. */
                    assert!(
                        hits.iter().any(|h| h.text == rtext),
                        "op {op} I3 SELF-RECALL: live record {id} ('{rtext}', s{rstrength}) missing from its own top-8: {:?} — same-text rows: {:?}",
                        hits.iter().map(|h| (h.id, &h.text)).collect::<Vec<_>>(),
                        cma
                            .records
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.text == rtext)
                            .map(|(i, r)| (i, r.tier, r.entity_name.clone(), r.death_reason.clone()))
                            .collect::<Vec<_>>(),
                    );
                }
                }
            }
            // ── mark_positive (I7: counters only, never strength) ───
            3 => {
                if !cma.records.is_empty() {
                    let ids: Vec<u32> = (0..4)
                        .map(|_| rng.usize(cma.records.len()) as u32)
                        .collect();
                    let before: Vec<(u32, f32)> =
                        cma.records.iter().map(|r| (r.id, r.strength)).collect();
                    cma.mark_positive(&ids);
                    for (id, s0) in &before {
                        assert_eq!(
                            cma.records[*id as usize].strength,
                            *s0,
                            "op {op} I7: mark_positive moved strength of {id} ({}→{})",
                            s0,
                            cma.records[*id as usize].strength
                        );
                    }
                }
            }
            // ── forget (never pins/entities — keeps I8 decidable) ───
            4 => {
                if !cma.records.is_empty() {
                    let id = rng.usize(cma.records.len()) as u32;
                    let rec = &cma.records[id as usize];
                if rec.role != PIN_ROLE && rec.tier != super::cma::TIER_ENTITY {
                    let t = rec.text.clone();
                    if cma.forget(id) {
                        // forget() is ROW-level: a same-text sibling may
                        // live in records (dup-gate miss) or in the
                        // working slot — the text only joins the dead set
                        // when neither survives.
                        let still_live = cma
                            .records
                            .iter()
                            .any(|r| r.tier != TIER_FORGOTTEN && r.text == t)
                            || cma.working.iter().any(|w| w.text == t);
                        if !still_live {
                            dead.insert(t);
                        }
                    }
                }
                }
            }
            // ── decay ───────────────────────────────────────────────
            5 => {
                let dt = (1 + rng.usize(7 * 86400)) as u32;
                cma.decay_tick(dt);
            }
            // ── retain (I2 capture + I8) ────────────────────────────
            6 => {
                let _ = cma.retain();
                capture_dead(&cma, &snapshot_live, &mut dead, op);
                // I8: every unforgotten pin and every entity still stands
                // (identity by role/name — retain renumbers ids, so a
                // stored id proves nothing)
                for text in &pins {
                    assert!(
                        cma
                            .records
                            .iter()
                            .any(|r| r.role == PIN_ROLE && r.text == *text && r.tier != TIER_FORGOTTEN),
                        "op {op} I8: pin '{text}' did not survive retain — rows carrying the text: {:?}, total pin-role rows: {}",
                        cma
                            .records
                            .iter()
                            .filter(|r| r.text == *text)
                            .map(|r| (r.id, &r.role, r.tier, r.strength))
                            .collect::<Vec<_>>(),
                        cma.records.iter().filter(|r| r.role == PIN_ROLE).count(),
                    );
                }
                for name in entities.keys() {
                    assert!(
                        cma.records.iter().any(|r| r.entity_name.as_deref() == Some(name.as_str())),
                        "op {op} I8: entity '{name}' did not survive retain"
                    );
                }
            }
            // ── consolidate (rare) ──────────────────────────────────
            7 => {
                if rng.usize(32) == 0 {
                    let _ = cma.consolidate();
                    capture_dead(&cma, &snapshot_live, &mut dead, op);
                }
            }
            // ── entity upsert ───────────────────────────────────────
            8 => {
                let k = rng.usize(ENTITY_POOL);
                let name = format!("ent-{k}");
                let got = cma.remember_entity(
                    &vec_for(TEXT_POOL + k),
                    &format!("entity {k} fact"),
                    &name,
                    "fuzz-entity",
                );
                assert!(got >= 0, "op {op}: entity upsert failed");
                entities.insert(name, got as u32);
            }
            // ── working churn + promotion (I5 / I6) ─────────────────
            _ => {
                let i = rng.usize(512);
                // Keys 3040..3551: inside the injective band, disjoint from
                // fact (0..2999) and entity (3000..3039) keys.
                let id = cma.remember_working(
                    &vec_for(TEXT_POOL + ENTITY_POOL + i),
                    &format!("working-{i}"),
                    "user",
                    0,
                    "fuzz",
                );
                // An explicit remember() of that text is a legal
                // resurrection if a promoted copy was tombstoned earlier.
                dead.remove(&format!("working-{i}"));
                working_ids.push(id);
                assert!(
                    cma.working.len() <= WORKING_MAX_RECORDS,
                    "op {op} I5: working set {} exceeds cap {WORKING_MAX_RECORDS}",
                    cma.working.len()
                );
                if !cma.working.is_empty() && rng.usize(4) == 0 {
                    // I6 on exactly ONE working record — the fresh/merged
                    // outcome stays decidable.
                    let w = &cma.working[0];
                    let (wid, created, src) = (w.id, w.created_at, w.source.clone());
                    let before = cma.len();
                    let (promoted, merged) = cma.promote_working_by_ids(&[wid]);
                    if promoted == 1 {
                        assert_eq!(
                            cma.len(),
                            before + 1,
                            "op {op} I6: promoted=1 but store grew by {}",
                            cma.len() - before
                        );
                        let r = &cma.records[before];
                        assert_eq!(
                            (r.created_at, r.source.as_str(), r.tier),
                            (created, src.as_str(), super::cma::TIER_EPISODIC),
                            "op {op} I6: promotion broke provenance"
                        );
                    } else {
                        assert_eq!(
                            (promoted, merged, cma.len()),
                            (0, 1, before),
                            "op {op} I6: outcome {}+{} with store {}→{}",
                            promoted,
                            merged,
                            before,
                            cma.len()
                        );
                    }
                }
                if rng.usize(12) == 0 {
                    cma.clear_working();
                    working_ids.clear();
                }
            }
        }

        verify_structure(&cma, op);



        // I4: periodic flush→reload
        if cma.len() > 0 && op % FLUSH_PERIOD == FLUSH_PERIOD / 2 {
            let snap: Vec<(f32, String, u32, String)> = cma
                .records
                .iter()
                .map(|r| (r.strength, r.text.clone(), r.created_at, r.source.clone()))
                .collect();
            let n = cma.len();
            let mut vecs = Vec::with_capacity(n * DIM);
            for idx in 0..n {
                vecs.extend_from_slice(cma.index.vector(idx as u32));
            }
            let json = cma.records_json();
            let mut reloaded = Cma::new(DIM);
            assert!(
                reloaded.hydrate(&vecs, n, DIM, &json),
                "op {op} I4: hydrate refused the flush payload"
            );
            assert_eq!(reloaded.len(), n, "op {op} I4: reload count drifted");
            for (idx, ((s0, t0, c0, src0), r)) in
                snap.iter().zip(reloaded.records.iter()).enumerate()
            {
                assert!(
                    (r.strength - s0).abs() < 1e-5 && &r.text == t0 && r.created_at == *c0,
                    "op {op} I4: record {idx} drifted ({}→{} / {t0:?}→{:?})",
                    s0,
                    r.strength,
                    &r.text
                );
                assert_eq!(
                    &r.source, src0,
                    "op {op} I4: provenance drifted on record {idx}"
                );
            }
            cma = reloaded;
            // working never serializes — the reload drops it by contract
            working_ids.clear();
            for w in cma.working.iter() {
                working_ids.push(w.id);
            }
        }
        if !working_ids.is_empty() {
            working_ids.retain(|w| cma.working.iter().any(|x| x.id == *w));
        }
    }
}

/// After retain/consolidation: any previously-live text that vanished
/// (tier-flipped or physically pruned) joins the dead set — unless a
/// same-text row still lives OUTSIDE records (the working slot), which
/// the records-only diff cannot see.
fn capture_dead(
    cma: &Cma,
    snapshot_live: &HashSet<String>,
    dead: &mut HashSet<String>,
    _op: usize,
) {
    let now: HashSet<String> = cma
        .records
        .iter()
        .filter(|r| r.tier != TIER_FORGOTTEN)
        .map(|r| r.text.clone())
        .collect();
    for t in snapshot_live.difference(&now) {
        if !cma.working.iter().any(|w| &w.text == t) {
            dead.insert(t.clone());
        }
    }
}

fn texts_equal(cma: &Cma, id: usize, t: &str) -> bool {
    cma.records.get(id).map(|r| r.text == t).unwrap_or(false)
}

fn verify_structure(cma: &Cma, op: usize) {
    // I1: dense ids
    for (idx, rec) in cma.records.iter().enumerate() {
        assert_eq!(
            rec.id as usize, idx,
            "op {op} I1: record {idx} carries id {}",
            rec.id
        );
    }
    // I5: working cap
    assert!(
        cma.working.len() <= WORKING_MAX_RECORDS,
        "op {op} I5: working set {} exceeds cap",
        cma.working.len()
    );
    // I9: the default list view never shows a forgotten-tier row
    let view = cma.list_json(&ListFilter {
        limit: 0, // 0 = unlimited (the documented contract)
        ..Default::default()
    });
    let v: serde_json::Value = serde_json::from_str(&view).expect("list_json is JSON");
    for rec in v["records"].as_array().expect("records array") {
        assert_ne!(
            rec["tier"].as_u64(),
            Some(TIER_FORGOTTEN as u64),
            "op {op} I9: default view leaked a tombstone: {}",
            rec["text"]
        );
    }
}
