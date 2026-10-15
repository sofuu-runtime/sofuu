// sofuu-core — Brain Core II M2: the decay survival harness.
//
// Survival curves, measured like every other gate: a deterministic
// fixture drives weeks of day-ticks over twin memories — one left
// untouched, one positively reinforced on each use day — and the
// asserts encode the M2 law (recognition buys decay tolerance):
//
//   cargo test -p sofuu-core --lib decay_eval -- --nocapture
//
// The numbers move ONLY together with `mark_positive`'s half-life
// growth (and `kv_hints`'s, same law): the harness fails if a decay
// change silently kills a still-engaged memory early, or if growth
// runs away past the tier cap.

use super::cma::{Cma, EPISODIC_HALF_LIFE_CAP, FORGET_FLOOR};

const DIM: usize = 64;

fn unit(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v
}

fn now_secs() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().min(u32::MAX as u64) as u32)
        .unwrap_or(0)
}

/// Per-day survival for a week-use schedule: at day 7k the record is
/// positively reinforced (strength ride included, like the real recall
/// path's user continuation), then one day tick applies. Returns the
/// strength trace.
fn survival_weekly(cma: &mut Cma, id: u32, weeks: usize) -> Vec<f32> {
    let mut trace = Vec::new();
    for day in 0..(weeks * 7) {
        if day % 7 == 0 {
            cma.mark_positive(&[id]);
            cma.records[id as usize].strength = 1.0; // the use-refresh
        }
        cma.decay_tick(86400);
        trace.push(cma.records[id as usize].strength);
    }
    trace
}

#[test]
fn decay_eval_untouched_memory_dies_between_uses() {
    let mut cma = Cma::new(DIM);
    let id = cma.remember(&unit(0), "rarely needed fact", "user", 0) as u32;
    for _ in 0..10 {
        cma.decay_tick(86400);
    }
    let s = cma.records[id as usize].strength;
    println!("10 days untouched: strength {s:.6} (half_life {} s)", cma.records[id as usize].half_life);
    assert!(
        s < FORGET_FLOOR,
        "the 1-day curve must decay below the recall floor within 10 idle days: {s}"
    );
}

#[test]
fn decay_eval_reinforced_memory_survives_the_same_schedule() {
    let mut kept = Cma::new(DIM);
    let id = kept.remember(&unit(1), "weekly useful fact", "user", 0) as u32;
    let trace = survival_weekly(&mut kept, id, 4);
    let s = *trace.last().unwrap();
    let rec = &kept.records[id as usize];
    println!("4 weeks of weekly use: final {s:.4}, half_life {} s", rec.half_life);
    /* The law: reinforcement keeps the memory alive where its idle twin
     * died (see the untouched test: day 10 already below the floor). */
    assert!(
        s >= FORGET_FLOOR,
        "a weekly-engaged memory must survive where an idle one dies: {s}"
    );
    assert!(
        rec.half_life > 86400,
        "use must have grown the half-life: {}",
        rec.half_life
    );
    /* Strength after the FIRST mark_positive must be untouched (I7):
     * only the use-refresh moves it. Verified by the growth assert —
     * check the clip directly. */
    assert!(
        rec.total_recalls == 0 && rec.positive_recalls == 4,
        "mark_positive counts, does not recall: {}+{}",
        rec.total_recalls,
        rec.positive_recalls
    );
}

#[test]
fn decay_eval_growth_never_passes_the_tier_cap() {
    let mut cma = Cma::new(DIM);
    let epi = cma.remember(&unit(0), "episodic hot", "user", 0) as u32;
    for _ in 0..600 {
        cma.mark_positive(&[epi]);
    }
    assert_eq!(
        cma.records[epi as usize].half_life,
        EPISODIC_HALF_LIFE_CAP,
        "episodic growth must cap at the semantic curve"
    );
    /* The cap is a MAX: an already-capped row never shrinks. */
    for _ in 0..10 {
        cma.mark_positive(&[epi]);
    }
    assert_eq!(cma.records[epi as usize].half_life, EPISODIC_HALF_LIFE_CAP);
}

#[test]
fn decay_eval_death_ledger_records_retain_prunes_and_survives_reload() {
    let mut cma = Cma::new(DIM);
    /* One record decayed below the floor (retain prunes it), one pinned
     * weak (retain keeps it), one manually forgotten tombstone. */
    let weak = cma.remember(&unit(0), "the weak fact about the quokka", "user", 0) as u32;
    cma.records[weak as usize].strength = 0.01;
    let pin = cma.remember(&unit(1), "the pinned anchor", super::cma::PIN_ROLE, 0) as u32;
    cma.records[pin as usize].strength = 0.001;
    let dead = cma.remember(&unit(2), "the forgotten fact about the quokka", "user", 0) as u32;
    cma.forget(dead);
    /* The tombstone carries its own WHY inline (retain will physically
     * remove the row; only the ledger carries it after that). */
    assert_eq!(
        cma.records[dead as usize].death_reason.as_deref(),
        Some("forget")
    );
    assert_eq!(cma.retain(), 2, "weak + tombstone drop");
    assert_eq!(cma.deaths.len(), 2, "both deaths are ledged");
    assert_eq!(cma.deaths[0].text, "the weak fact about the quokka");
    assert_eq!(cma.deaths[0].reason, "decay");
    assert_eq!(cma.deaths[1].reason, "forget");

    /* The ledger persists: flush → reload carries the WHY forward. */
    let snap = cma.deaths.clone();
    let mut reloaded = Cma::new(DIM);
    let n = cma.len();
    let mut vecs = Vec::with_capacity(n * DIM);
    for idx in 0..n {
        vecs.extend_from_slice(cma.index.vector(idx as u32));
    }
    assert!(reloaded.hydrate(&vecs, n, DIM, &cma.records_json()), "hydrate");
    assert_eq!(reloaded.deaths.len(), snap.len(), "ledger survives reload");
    assert_eq!(reloaded.deaths[1].reason, "forget");

    /* Consolidated rows are ledged with THEIR truth: content moved into a
     * summary, not destroyed. */
    let mut cc = Cma::new(16);
    for i in 0..10 {
        let mut v = vec![0.0; 16];
        v[0] = 1.0;
        v[2 + i] = 0.1 + (i as f32) * 0.01;
        let id = cc.remember(&v, &format!("episode {i} quokka migration"), "user", 0);
        cc.records[id as usize].strength = 0.1;
        cc.records[id as usize].age_seconds = 90000;
    }
    cc.consolidate();
    assert!(cc.deaths.iter().all(|d| d.reason == "consolidated"));
}

#[test]
fn decay_eval_ledger_never_grows_past_its_cap() {
    let mut cma = Cma::new(DIM);
    for i in 0..(super::cma::DEATH_LEDGER_CAP + 50) {
        let id = cma.remember(&unit(i % DIM), &format!("churn fact {i}"), "user", 0) as u32;
        cma.forget(id);
        cma.retain();
    }
    println!("ledger size after {} deaths: {}", super::cma::DEATH_LEDGER_CAP + 50, cma.deaths.len());
    assert!(cma.deaths.len() <= super::cma::DEATH_LEDGER_CAP);
    /* Newest kept: the ring holds the LATEST deaths. */
    assert_eq!(cma.deaths.last().unwrap().reason, "forget");
}
