// ml/online.rs -- online adaptation of the supervisor's output layer
// (PLAN-ML-GATES §13).
//
// The pretrained supervisor is the starting point, not the finished
// product: real runs produce outcome labels (errored / near-empty results
// are the waste proxy) and the user can mark a nudge wrong (/ml wrong) or
// a missed call wasteful (/ml wasted). This module turns that feedback
// into a re-fit of the OUTPUT LAYER ONLY, under guardrails that keep the
// adaptation from becoming its own failure mode:
//
//   1. output layer only -- the hidden layers are frozen (net.rs
//      out_start); the 45 weights of W3+b3 are the only slice that moves;
//   2. trust region -- adapted weights are projected back into the ball
//      around the pretrained layer (clamp_trust_region), bounded by
//      construction, not by hoping the learning rate is small; every
//      learn() re-fits FROM the pretrained layer, so drift can never
//      accumulate across rounds;
//   3. replay anchors -- every adaptation batch includes the canonical
//      runtime fixtures, so the layer cannot forget what the acceptance
//      gates proved;
//   4. batch floor -- nothing adapts on fewer than MIN_EXAMPLES labeled
//      examples (a 3-example fit is overfitting by construction);
//   5. adoption gate -- the candidate must classify ALL fixtures
//      correctly at the shipped threshold: the hand-written out-of-domain
//      anchors PLUS the per-family fixtures baked from the training
//      distribution (supervisor/fixtures_v1.f32, emitted by ml-train);
//   6. confidence-weighted labels -- proxy labels carry the confidence of
//      the signal that produced them, and explicit user corrections
//      (/ml wrong, /ml wasted) outweigh proxies (GOLD_WEIGHT);
//   7. two-step adoption -- learn() only TRAINS a candidate and holds it
//      pending (in-memory, never persisted unadopted); /ml adopt applies
//      it, /ml discard drops it. A human stays in the loop;
//   8. separate persistence -- ~/.sofuu/ml/supervisor_online.f32, keyed
//      by the FNV-1a hash of the pretrained blob: a re-bake changes the
//      hash and silently invalidates every stored adaptation;
//   9. off by default -- observations accumulate always (they are cheap),
//      but nothing adapts until /ml learn says so.
//
// Advise-only still (Principle 1): an adapted layer changes which nudges
// fire, never what the data path carries.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};

use crate::ml::net::clamp_trust_region;
use crate::ml::supervisor::features::{self, SupervisorContext, TrajCall};
use crate::ml::supervisor::model;

/// Batch floor (§13 guardrail 4).
pub const MIN_EXAMPLES: usize = 16;

const MAX_OBS: usize = 512;
const MAX_EXAMPLES: usize = 1024;
const BATCH_CAP: usize = 256;
/// Trust-region radius, relative to the pretrained layer's norm
/// (§13 guardrail 2).
const TRUST_REL: f32 = 0.25;
const LEARN_LR: f32 = 0.05;
const LEARN_EPOCHS: usize = 12;
/// Explicit user corrections (/ml wrong, /ml wasted) weigh double the
/// strongest proxy label -- they are ground truth, the proxies estimates.
const GOLD_WEIGHT: f32 = 2.0;
/// Upper bound for any single example weight (proxy confidences arrive
/// clamped to this on the way in).
const MAX_WEIGHT: f32 = 2.0;

const ONLINE_MAGIC: u32 = 0x314E_4F53; // "SON1" little-endian
const ONLINE_VERSION: u32 = 1;

const FIXTURES_MAGIC: u32 = u32::from_le_bytes(*b"SFX1");
const FIXTURES_VERSION: u32 = 1;

/// One observed checkpoint: the feature vector as scored, plus the
/// verdict, awaiting an outcome label.
struct Obs {
    run: String,
    step: u32,
    feats: [f32; 33],
    flagged: bool,
    #[allow(dead_code)]
    score: f32,
}

/// A labeled example: features, the ground-truth waste label, and the
/// weight of the signal that produced the label (guardrail 6). The SGD
/// gradient is scaled by `w`, so a conf-0.6 proxy moves the layer 30%
/// less than a certain one, and a user correction moves it 2x more.
#[derive(Clone)]
struct Example {
    feats: [f32; 33],
    y: f32,
    w: f32,
}

struct OnlineState {
    enabled: bool,
    /// Checkpoints awaiting outcome labels (ring, bounded).
    obs: VecDeque<Obs>,
    /// Labeled examples ready for the next adaptation (bounded FIFO).
    examples: Vec<Example>,
    /// The adopted output layer (W3 then b3), None = pretrained.
    adapted: Option<Vec<f32>>,
    /// A trained candidate awaiting /ml adopt (guardrail 7). In-memory
    /// only -- never persisted before a human adopts it.
    pending: Option<Vec<f32>>,
    adaptations: u32,
}

static ONLINE: LazyLock<Mutex<OnlineState>> = LazyLock::new(|| {
    let mut st = OnlineState {
        enabled: false,
        obs: VecDeque::new(),
        examples: Vec::new(),
        adapted: None,
        pending: None,
        adaptations: 0,
    };
    st.adapted = load_persisted();
    Mutex::new(st)
});

fn with_online<R>(f: impl FnOnce(&mut OnlineState) -> R) -> R {
    let mut st = ONLINE.lock().unwrap();
    f(&mut st)
}

/* ── Observation + labeling ─────────────────────────────────────────── */

/// Record one checkpoint (called by supervisor::model::check). Cheap and
/// always on -- observations are inert until /ml learn enables adaptation.
pub fn observe(run: &str, step: u32, feats: &[f32; 33], flagged: bool, score: f32) {
    with_online(|st| {
        if st.obs.len() >= MAX_OBS {
            st.obs.pop_front();
        }
        st.obs.push_back(Obs {
            run: run.to_string(),
            step,
            feats: *feats,
            flagged,
            score,
        });
    });
}

/// Outcome label from the agent loop ({kind:"outcome"} on sofuu.ml.feedback).
/// `conf` is the confidence of the proxy that produced the label (the agent
/// abstains entirely on ambiguous results rather than send a coin-flip
/// label). Returns true when the checkpoint was found and labeled.
pub fn label_outcome(run: &str, step: u32, wasted: bool, conf: f32) -> bool {
    let w = conf.clamp(0.0, MAX_WEIGHT);
    with_online(|st| {
        let idx = st
            .obs
            .iter()
            .position(|o| o.run == run && o.step == step);
        let Some(idx) = idx else { return false };
        let o = st.obs.remove(idx).unwrap();
        push_example(st, Example { feats: o.feats, y: if wasted { 1.0 } else { 0.0 }, w });
        true
    })
}

/// User feedback ({kind:"wrong"}): the most recent flag was a mistake --
/// the call was NOT waste. Gold label: outweighs proxy labels. Returns
/// true when a flagged checkpoint existed.
pub fn label_wrong() -> bool {
    with_online(|st| {
        let idx = st.obs.iter().rposition(|o| o.flagged);
        let Some(idx) = idx else { return false };
        let o = st.obs.remove(idx).unwrap();
        push_example(st, Example { feats: o.feats, y: 0.0, w: GOLD_WEIGHT });
        true
    })
}

/// User feedback ({kind:"wasted"}): the most recent call WAS waste even
/// though the supervisor let it through -- the mirror of label_wrong, so
/// missed waste is correctable too, not just false flags. Gold label.
pub fn label_wasted() -> bool {
    with_online(|st| {
        if st.obs.is_empty() {
            return false;
        }
        let o = st.obs.pop_back().unwrap();
        push_example(st, Example { feats: o.feats, y: 1.0, w: GOLD_WEIGHT });
        true
    })
}

fn push_example(st: &mut OnlineState, ex: Example) {
    if st.examples.len() >= MAX_EXAMPLES {
        st.examples.remove(0);
    }
    st.examples.push(ex);
}

/* ── Runtime fixtures: replay anchors + adoption gate ───────────────── */

/// Canonical cases in a domain no training family touches (wind-turbine
/// governor). They anchor every adaptation batch (guardrail 3) and gate
/// every candidate (guardrail 5): an adaptation that regresses ANY of
/// these is discarded, whatever the live examples say.
fn fixtures() -> Vec<Example> {
    const TASK: &str = "tune the wind turbine governor before the winter storms";
    const P0: &str = "src/turbine/governor.rs";
    const P1: &str = "src/turbine/pitch.rs";

    fn tc(tool: &str, sig: &str, target: &str, chars: u32) -> TrajCall {
        TrajCall { tool: tool.to_string(), sig: sig.to_string(), target: target.to_string(), result_chars: chars, errored: false }
    }
    fn feats(task: &str, calls: Vec<TrajCall>, tool: &str, args: &str, target: &str) -> [f32; 33] {
        let step = calls.len() as u32 + 1; // the check runs at the NEXT call
        let sig = if tool == "__loop__" { String::new() } else { format!("{tool}:{args}") };
        let ctx = SupervisorContext {
            task: task.to_string(),
            budget: 20,
            calls,
            skip_targets: Vec::new(),
            tool: tool.to_string(),
            args_text: args.to_string(),
            target: target.to_string(),
            sig,
            step,
        };
        features::extract(&ctx)
    }

    let read0 = format!("{{\"path\":\"{P0}\"}}");
    let r0 = tc("read_file", &format!("read_file:{read0}"), P0, 1200);
    let grep_gain = tc("grep", "{\"pattern\":\"governor_gain\"}", "governor_gain", 500);

    vec![
        // waste: exact duplicate of the call just made
        Example { feats: feats(TASK, vec![r0.clone()], "read_file", &read0, P0), y: 1.0, w: 1.0 },
        // waste: re-read of an unchanged file (offset args)
        Example { feats: feats(TASK, vec![r0.clone()], "read_file", &format!("{{\"offset\":50,\"path\":\"{P0}\"}}"), P0), y: 1.0, w: 1.0 },
        // waste: degenerate broad search
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\".\"}", "."), y: 1.0, w: 1.0 },
        // waste: off-task wander after an on-task read
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\"sourdough starter feeding schedule\"}", "sourdough starter feeding schedule"), y: 1.0, w: 1.0 },
        // clean: first call reads the task's core file
        Example { feats: feats(TASK, vec![], "read_file", &read0, P0), y: 0.0, w: 1.0 },
        // clean: follow-up grep for a symbol from the file just read
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\"governor_gain\"}", "governor_gain"), y: 0.0, w: 1.0 },
        // clean: similar-but-different second symbol
        Example { feats: feats(TASK, vec![r0.clone(), grep_gain.clone()], "grep", "{\"pattern\":\"pitch_trim\"}", "pitch_trim"), y: 0.0, w: 1.0 },
        // clean: a new on-task file after a write
        Example { feats: feats(TASK, vec![r0.clone(), tc("edit_file", &format!("{{\"path\":\"{P0}\"}}"), P0, 60)], "read_file", &format!("{{\"path\":\"{P1}\"}}"), P1), y: 0.0, w: 1.0 },
    ]
}

static FIXTURES: LazyLock<Vec<Example>> = LazyLock::new(fixtures);

static FIXTURES_BLOB: &[u8] = include_bytes!("supervisor/fixtures_v1.f32");

/// Per-family fixtures baked from the training distribution (emitted by
/// ml-train after the acceptance gates pass): one margin-separated
/// representative per construction family. Format:
///
///   "SFX1" | version u32 | count u32 | crc32 u32 | (y f32 + 33×f32)·count
///
/// Any mismatch refuses the whole blob -- the gate then rests on the
/// hand-written anchors alone (a smaller net, never a wrong one). The
/// cargo test below asserts the committed blob parses in full, so a
/// corrupt bake fails CI, not the user.
fn parse_baked_fixtures(bytes: &[u8]) -> Vec<Example> {
    if bytes.len() < 16 {
        return Vec::new();
    }
    let mut hdr = [0u32; 4];
    for (i, h) in hdr.iter_mut().enumerate() {
        *h = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let [magic, version, count, crc] = hdr;
    if magic != FIXTURES_MAGIC || version != FIXTURES_VERSION {
        return Vec::new();
    }
    let payload_len = count as usize * 34 * 4;
    if bytes.len() != 16 + payload_len {
        return Vec::new();
    }
    let payload = &bytes[16..];
    if crate::ml::net::crc32_ieee(payload) != crc {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for chunk in payload.chunks_exact(34 * 4) {
        let y = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let mut feats = [0f32; 33];
        for (i, slot) in feats.iter_mut().enumerate() {
            *slot = f32::from_le_bytes(chunk[4 + i * 4..8 + i * 4].try_into().unwrap());
        }
        out.push(Example { feats, y, w: 1.0 });
    }
    out
}

static BAKED_FIXTURES: LazyLock<Vec<Example>> = LazyLock::new(|| parse_baked_fixtures(FIXTURES_BLOB));

/* ── Adaptation ─────────────────────────────────────────────────────── */

pub struct LearnReport {
    pub enabled: bool,
    pub examples: usize,
    /// A candidate was trained, passed the gate, and awaits /ml adopt.
    pub pending: bool,
    pub gate_passed: usize,
    pub gate_total: usize,
    /// The trust region had to project the candidate back.
    pub clamped: bool,
    pub detail: String,
}

/// Enable online learning and (when enough labels exist) train a
/// CANDIDATE output layer. Two-step by design (guardrail 7): the
/// candidate is held pending in memory; /ml adopt applies it, /ml
/// discard drops it. Idempotent enable; called by /ml learn.
pub fn learn() -> LearnReport {
    with_online(|st| {
        st.enabled = true;
        let n = st.examples.len();
        if n < MIN_EXAMPLES {
            return LearnReport {
                enabled: true,
                examples: n,
                pending: false,
                gate_passed: 0,
                gate_total: 0,
                clamped: false,
                detail: format!("need at least {MIN_EXAMPLES} labeled examples, have {n}"),
            };
        }

        // Batch: the most recent live examples + the replay anchors.
        // The baked per-family fixtures gate the candidate but stay out
        // of the batch so anchors don't dilute the live signal.
        let start = n.saturating_sub(BATCH_CAP);
        let mut batch: Vec<Example> = st.examples[start..].to_vec();
        batch.extend(FIXTURES.iter().cloned());

        // Candidate: the pretrained output layer, re-fit by plain SGD on
        // the confidence-weighted logistic loss. Deterministic: fixed
        // batch order, scalar f32. Always starts FROM the pretrained
        // layer, so drift never accumulates across learn() rounds.
        let net = model::net();
        let pretrained = net.output_layer().to_vec();
        let h2 = net.h2 as usize;
        let mut w3 = pretrained.clone();
        for _ in 0..LEARN_EPOCHS {
            for ex in batch.iter() {
                let h = net.forward_hidden(&ex.feats);
                let p = net.forward_output_with(&w3, &h);
                let g = ex.w * (p - ex.y);
                for j in 0..h2 {
                    w3[j] -= LEARN_LR * g * h[j];
                }
                w3[h2] -= LEARN_LR * g;
            }
        }
        let clamped = clamp_trust_region(&mut w3, &pretrained, TRUST_REL);

        // Adoption gate (guardrail 5): the candidate must keep EVERY
        // fixture correct at the shipped threshold -- the hand-written
        // out-of-domain anchors plus the baked per-family fixtures.
        let gate: Vec<&Example> = FIXTURES.iter().chain(BAKED_FIXTURES.iter()).collect();
        let mut passed = 0usize;
        for ex in gate.iter() {
            let h = net.forward_hidden(&ex.feats);
            let p = net.forward_output_with(&w3, &h);
            let flagged = p >= model::THRESHOLD;
            if flagged == (ex.y >= 0.5) {
                passed += 1;
            }
        }
        if passed < gate.len() {
            return LearnReport {
                enabled: true,
                examples: n,
                pending: false,
                gate_passed: passed,
                gate_total: gate.len(),
                clamped,
                detail: format!(
                    "candidate regressed {} of {} adoption fixtures -- discarded",
                    gate.len() - passed,
                    gate.len()
                ),
            };
        }

        st.pending = Some(w3);
        LearnReport {
            enabled: true,
            examples: n,
            pending: true,
            gate_passed: passed,
            gate_total: gate.len(),
            clamped,
            detail: format!(
                "candidate re-fit on {n} examples; {passed}/{} adoption fixtures pass; trust region {} -- /ml adopt to apply, /ml discard to drop",
                gate.len(),
                if clamped { "applied" } else { "not needed" }
            ),
        }
    })
}

/// Apply the pending candidate (guardrail 7, step 2). Returns
/// (adopted, persisted). Called by /ml adopt.
pub fn adopt() -> (bool, bool) {
    with_online(|st| {
        let Some(w3) = st.pending.take() else { return (false, false) };
        st.adapted = Some(w3);
        st.adaptations += 1;
        (true, persist(st.adapted.as_deref().unwrap()))
    })
}

/// Drop the pending candidate without applying it. Returns true when a
/// candidate existed. Called by /ml discard.
pub fn discard() -> bool {
    with_online(|st| st.pending.take().is_some())
}

/// The adopted output layer, if any (cloned -- h2+1 floats, cheap).
pub fn output_layer() -> Option<Vec<f32>> {
    with_online(|st| st.adapted.clone())
}

/// Drop the adaptation AND any pending candidate, clear the label buffer,
/// remove the persisted blob. Called by /ml reset.
pub fn reset() {
    with_online(|st| {
        st.adapted = None;
        st.pending = None;
        st.examples.clear();
        st.adaptations = 0;
    });
    let _ = std::fs::remove_file(persist_path());
}

/// (enabled, observations, labeled examples, adapted, adaptations, pending)
pub fn status() -> (bool, usize, usize, bool, u32, bool) {
    with_online(|st| {
        (
            st.enabled,
            st.obs.len(),
            st.examples.len(),
            st.adapted.is_some(),
            st.adaptations,
            st.pending.is_some(),
        )
    })
}

/* ── Persistence (§13 guardrail 8) ──────────────────────────────────── */

fn persist_path() -> std::path::PathBuf {
    let mut p = std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    p.push(".sofuu");
    p.push("ml");
    p.push("supervisor_online.f32");
    p
}

/// "SON1" | version | blob-hash (2×u32) | h2 | n | crc32 | weights f32le
fn persist(w3: &[f32]) -> bool {
    let path = persist_path();
    if let Some(dir) = path.parent() {
        if std::fs::create_dir_all(dir).is_err() {
            return false;
        }
    }
    let hash = model::blob_hash();
    let h2 = (w3.len().saturating_sub(1)) as u32;
    let mut weight_bytes = Vec::with_capacity(w3.len() * 4);
    for v in w3 {
        weight_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let crc = crate::ml::net::crc32_ieee(&weight_bytes);
    let mut out = Vec::with_capacity(28 + weight_bytes.len());
    for v in [
        ONLINE_MAGIC,
        ONLINE_VERSION,
        hash as u32,
        (hash >> 32) as u32,
        h2,
        w3.len() as u32,
        crc,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&weight_bytes);
    std::fs::write(&path, &out).is_ok()
}

/// Load the persisted adaptation IFF it was keyed to the CURRENT pretrained
/// blob. A re-bake (or any corruption) yields None -- the pretrained layer
/// is the fallback, never a stale adaptation.
fn load_persisted() -> Option<Vec<f32>> {
    let bytes = std::fs::read(persist_path()).ok()?;
    if bytes.len() < 28 {
        return None;
    }
    let mut hdr = [0u32; 7];
    for (i, h) in hdr.iter_mut().enumerate() {
        *h = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().ok()?);
    }
    let [magic, version, hash_lo, hash_hi, h2, n, crc] = hdr;
    if magic != ONLINE_MAGIC || version != ONLINE_VERSION {
        return None;
    }
    let hash = (hash_lo as u64) | ((hash_hi as u64) << 32);
    if hash != model::blob_hash() {
        return None; // pretrained weights changed -- adaptation is stale
    }
    if n != h2 + 1 || bytes.len() != 28 + n as usize * 4 {
        return None;
    }
    let weight_bytes = &bytes[28..];
    if crate::ml::net::crc32_ieee(weight_bytes) != crc {
        return None;
    }
    Some(
        weight_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

/// The ONLINE state is process-global and cargo runs tests on parallel
/// threads, so every test that mutates or reads it observably serializes
/// here: the online tests below AND the supervisor eval tests, whose
/// check()/loop_check() calls record observations into the same deque and
/// score against whatever output layer is live at that instant.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_separated_by_the_pretrained_net() {
        // The adoption gate is only meaningful if the pretrained layer
        // itself classifies every anchor correctly -- hand-written AND
        // baked (ml-train filters the baked set by margin, this asserts
        // the committed blob survived that filter).
        let net = model::net();
        for (i, ex) in FIXTURES.iter().enumerate() {
            let p = net.forward_output(&net.forward_hidden(&ex.feats));
            let flagged = p >= model::THRESHOLD;
            assert_eq!(
                flagged,
                ex.y >= 0.5,
                "hand-written fixture {i} misclassified by pretrained net (p={p:.3})"
            );
        }
        for (i, ex) in BAKED_FIXTURES.iter().enumerate() {
            let p = net.forward_output(&net.forward_hidden(&ex.feats));
            let flagged = p >= model::THRESHOLD;
            assert_eq!(
                flagged,
                ex.y >= 0.5,
                "baked fixture {i} misclassified by pretrained net (p={p:.3})"
            );
        }
    }

    #[test]
    fn baked_fixtures_parse_from_the_committed_blob() {
        // One representative per training family: 23 families exist
        // (S0-S15, S17-S23); the trainer requires >= 18 survivors.
        assert!(
            BAKED_FIXTURES.len() >= 18,
            "expected >= 18 baked fixtures, got {}",
            BAKED_FIXTURES.len()
        );
        let waste = BAKED_FIXTURES.iter().filter(|e| e.y >= 0.5).count();
        assert!(waste > 0 && waste < BAKED_FIXTURES.len(), "both labels must be present");
    }

    #[test]
    fn observe_and_label_roundtrip() {
        let _g = TEST_LOCK.lock().unwrap();
        let feats = FIXTURES[0].feats;
        observe("online-test-run", 99, &feats, true, 0.9);
        assert!(label_outcome("online-test-run", 99, true, 1.0));
        assert!(!label_outcome("online-test-run", 99, true, 1.0), "consumed");
    }

    #[test]
    fn learn_refuses_small_batches() {
        let _g = TEST_LOCK.lock().unwrap();
        // Swap in a too-small example buffer, call learn() OUTSIDE the lock
        // (the mutex is not reentrant), then restore.
        let saved = with_online(|st| {
            let saved = st.examples.clone();
            st.examples.clear();
            for ex in FIXTURES.iter().take(3) {
                st.examples.push(ex.clone());
            }
            saved
        });
        let r = learn();
        with_online(|st| st.examples = saved);
        assert!(r.enabled);
        assert!(!r.pending);
        assert!(r.detail.contains("at least"));
    }

    #[test]
    fn label_wrong_needs_a_flagged_observation() {
        let _g = TEST_LOCK.lock().unwrap();
        with_online(|st| st.obs.clear());
        assert!(!label_wrong());
        observe("online-test-run", 7, &FIXTURES[3].feats, true, 0.9);
        assert!(label_wrong());
    }

    #[test]
    fn label_wasted_takes_the_most_recent_observation() {
        let _g = TEST_LOCK.lock().unwrap();
        with_online(|st| st.obs.clear());
        assert!(!label_wasted());
        observe("online-test-run", 1, &FIXTURES[4].feats, false, 0.1);
        observe("online-test-run", 2, &FIXTURES[5].feats, false, 0.2);
        assert!(label_wasted());
        // The consumed observation is gone: only step 1 remains.
        let left = with_online(|st| st.obs.iter().map(|o| o.step).collect::<Vec<_>>());
        assert_eq!(left, vec![1]);
    }

    #[test]
    fn zero_weight_examples_do_not_move_the_candidate() {
        // Confidence weighting is the guardrail against noisy proxies: a
        // w=0 example must leave the candidate bit-identical. Train twice
        // on the same anchor buffer, once with extra zero-weight examples
        // mixed in, and compare the pending layers.
        let _g = TEST_LOCK.lock().unwrap();
        let base: Vec<Example> = FIXTURES.iter().cloned().cycle().take(16).collect();
        let mut padded = base.clone();
        for ex in FIXTURES.iter().take(4) {
            let mut e = ex.clone();
            e.y = 1.0 - e.y; // hostile labels...
            e.w = 0.0; // ...with zero weight
            padded.push(e);
        }
        let train = |buf: Vec<Example>| -> Option<Vec<f32>> {
            let saved = with_online(|st| {
                let saved = st.examples.clone();
                st.examples = buf;
                st.pending = None;
                saved
            });
            let r = learn();
            let layer = with_online(|st| st.pending.take());
            with_online(|st| st.examples = saved);
            assert!(r.pending, "gate must pass on the anchor buffer: {}", r.detail);
            layer
        };
        let a = train(base).expect("candidate");
        let b = train(padded).expect("candidate");
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "zero-weight examples must not change the candidate"
        );
    }

    #[test]
    fn learn_holds_pending_until_adopt() {
        // Two-step adoption: learn() trains but does NOT activate; adopt()
        // activates + persists; reset() cleans up (removes the blob too).
        let _g = TEST_LOCK.lock().unwrap();
        let saved = with_online(|st| {
            let saved = (st.examples.clone(), st.adapted.clone(), st.adaptations);
            st.examples = FIXTURES.iter().cloned().collect(); // 8 >= floor? no -- 16 needed
            st.examples.extend(FIXTURES.iter().cloned()); // 16 total
            st.adapted = None;
            st.pending = None;
            saved
        });
        let r = learn();
        assert!(r.pending, "candidate must pass the gate on anchors: {}", r.detail);
        assert_eq!(r.gate_passed, r.gate_total);
        assert!(output_layer().is_none(), "learn() must not activate the candidate");

        let (adopted, _persisted) = adopt();
        assert!(adopted);
        assert!(output_layer().is_some(), "adopt() activates the candidate");
        assert!(!discard(), "nothing pending after adopt");

        reset();
        assert!(output_layer().is_none());
        // Restore the prior state for other tests.
        with_online(|st| {
            st.examples = saved.0;
            st.adapted = saved.1;
            st.adaptations = saved.2;
        });
    }
}
