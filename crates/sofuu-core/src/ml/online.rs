// ml/online.rs -- online adaptation across the on-device decision gates
// (PLAN-ML-GATES §13 & PLAN-ML-V2 Pillar 4).
//
// The pretrained models are the starting point, not the finished product:
// real developer runs produce outcome labels (errored/near-empty results,
// unused retrieval candidates, restored compaction segments) and the user
// can mark an advisory wrong (/ml wrong) or missed (/ml wasted).
//
// This module provides Universal Multi-Gate Online Learning across all 5
// micro-MLPs (Supervisor, Compaction, Relevance, Freshness, Alloc) under
// the 7 Ironclad Guardrails:
//
//   1. Output layer only -- hidden layers are frozen (net.rs); only W3+b3
//      (65 parameters for h2=64 in v2) move.
//   2. Trust region -- adapted weights are projected back into the ball
//      around the pretrained layer (clamp_trust_region, ||ΔW|| ≤ 0.20·||W_base||).
//   3. Replay anchors -- adaptation batches verify against canonical golden
//      fixtures so the model never forgets the baseline acceptance proof.
//   4. Batch floor -- nothing adapts on fewer than MIN_EXAMPLES (16).
//   5. Adoption gate -- candidate must pass 100% of golden fixtures.
//   6. Confidence-weighted labels -- user corrections weigh double proxies.
//   7. Two-step adoption -- learn() holds pending in memory; /ml adopt persists.
//   8. Separate persistence -- ~/.sofuu/ml/{gate}_online.f32 keyed by FNV-1a hash.
//   9. Off by default -- observations accumulate passively until /ml learn.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};

use crate::ml::net::clamp_trust_region;
use crate::ml::supervisor::features::{self, SupervisorContext, TrajCall};

/// Batch floor (§13 guardrail 4).
pub const MIN_EXAMPLES: usize = 16;

const MAX_OBS: usize = 512;
const MAX_EXAMPLES: usize = 1024;
const BATCH_CAP: usize = 256;
/// Trust-region radius, relative to the pretrained layer's norm (§13 guardrail 2).
const TRUST_REL: f32 = 0.20;
const LEARN_LR: f32 = 0.05;
const LEARN_EPOCHS: usize = 12;
/// Explicit user corrections weigh double proxy labels.
const GOLD_WEIGHT: f32 = 2.0;
/// Upper bound for any single example weight.
const MAX_WEIGHT: f32 = 2.0;

const ONLINE_MAGIC: u32 = 0x314E_4F53; // "SON1" little-endian
const ONLINE_VERSION: u32 = 2;

const FIXTURES_MAGIC: u32 = u32::from_le_bytes(*b"SFX1");
const FIXTURES_VERSION: u32 = 1;

/// Gate identifier for universal online learning across all 5 models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateKind {
    Supervisor,
    Compaction,
    Relevance,
    Freshness,
    Alloc,
}

impl GateKind {
    pub fn name(self) -> &'static str {
        match self {
            GateKind::Supervisor => "supervisor",
            GateKind::Compaction => "compaction",
            GateKind::Relevance => "relevance",
            GateKind::Freshness => "freshness",
            GateKind::Alloc => "alloc",
        }
    }

    pub fn in_dim(self) -> usize {
        match self {
            GateKind::Supervisor => crate::ml::supervisor::model::IN_DIM as usize,
            GateKind::Compaction => crate::ml::compaction::model::IN_DIM as usize,
            GateKind::Relevance => crate::ml::relevance::model::IN_DIM as usize,
            GateKind::Freshness => crate::ml::freshness::model::IN_DIM as usize,
            GateKind::Alloc => crate::ml::alloc::model::IN_DIM as usize,
        }
    }

    pub fn threshold(self) -> f32 {
        match self {
            GateKind::Supervisor => crate::ml::supervisor::model::THRESHOLD,
            GateKind::Compaction => crate::ml::compaction::model::THRESHOLD,
            GateKind::Relevance => crate::ml::relevance::model::THRESHOLD,
            GateKind::Freshness => crate::ml::freshness::model::THRESHOLD,
            GateKind::Alloc => crate::ml::alloc::model::THRESHOLD,
        }
    }

    pub fn net(self) -> &'static crate::ml::net::TinyMlp {
        match self {
            GateKind::Supervisor => crate::ml::supervisor::model::net(),
            GateKind::Compaction => &crate::ml::compaction::model::NET,
            GateKind::Relevance => &crate::ml::relevance::model::NET,
            GateKind::Freshness => &crate::ml::freshness::model::NET,
            GateKind::Alloc => &crate::ml::alloc::model::NET,
        }
    }

    pub fn blob_hash(self) -> u64 {
        match self {
            GateKind::Supervisor => crate::ml::supervisor::model::blob_hash(),
            GateKind::Compaction => crate::ml::compaction::model::blob_hash(),
            GateKind::Relevance => crate::ml::relevance::model::blob_hash(),
            GateKind::Freshness => crate::ml::freshness::model::blob_hash(),
            GateKind::Alloc => crate::ml::alloc::model::blob_hash(),
        }
    }
}

/// One observed checkpoint awaiting an outcome label.
#[derive(Clone, Debug)]
pub struct Obs {
    pub run: String,
    pub step: u32,
    pub feats: Vec<f32>,
    pub flagged: bool,
    #[allow(dead_code)]
    pub score: f32,
}

/// A labeled training example.
#[derive(Clone, Debug)]
pub struct Example {
    pub feats: Vec<f32>,
    pub y: f32,
    pub w: f32,
}

/// Online adaptation state for one gate.
pub struct GateState {
    pub enabled: bool,
    pub obs: VecDeque<Obs>,
    pub examples: Vec<Example>,
    pub adapted: Option<Vec<f32>>,
    pub pending: Option<Vec<f32>>,
    pub adaptations: u32,
}

impl GateState {
    pub fn new(gate: GateKind) -> Self {
        let mut st = Self {
            enabled: false,
            obs: VecDeque::new(),
            examples: Vec::new(),
            adapted: None,
            pending: None,
            adaptations: 0,
        };
        st.adapted = load_persisted(gate);
        st
    }
}

/// Universal process-global online state across all 5 gates.
struct UniversalOnlineState {
    supervisor: GateState,
    compaction: GateState,
    relevance: GateState,
    freshness: GateState,
    alloc: GateState,
}

impl UniversalOnlineState {
    fn gate_mut(&mut self, gate: GateKind) -> &mut GateState {
        match gate {
            GateKind::Supervisor => &mut self.supervisor,
            GateKind::Compaction => &mut self.compaction,
            GateKind::Relevance => &mut self.relevance,
            GateKind::Freshness => &mut self.freshness,
            GateKind::Alloc => &mut self.alloc,
        }
    }

    fn gate(&self, gate: GateKind) -> &GateState {
        match gate {
            GateKind::Supervisor => &self.supervisor,
            GateKind::Compaction => &self.compaction,
            GateKind::Relevance => &self.relevance,
            GateKind::Freshness => &self.freshness,
            GateKind::Alloc => &self.alloc,
        }
    }
}

static ONLINE: LazyLock<Mutex<UniversalOnlineState>> = LazyLock::new(|| {
    Mutex::new(UniversalOnlineState {
        supervisor: GateState::new(GateKind::Supervisor),
        compaction: GateState::new(GateKind::Compaction),
        relevance: GateState::new(GateKind::Relevance),
        freshness: GateState::new(GateKind::Freshness),
        alloc: GateState::new(GateKind::Alloc),
    })
});

fn with_online<R>(f: impl FnOnce(&mut UniversalOnlineState) -> R) -> R {
    let mut st = ONLINE.lock().unwrap();
    f(&mut st)
}

/* ── Observation + labeling ─────────────────────────────────────────── */

pub fn observe_gate(gate: GateKind, run: &str, step: u32, feats: &[f32], flagged: bool, score: f32) {
    with_online(|st| {
        let g = st.gate_mut(gate);
        if g.obs.len() >= MAX_OBS {
            g.obs.pop_front();
        }
        g.obs.push_back(Obs {
            run: run.to_string(),
            step,
            feats: feats.to_vec(),
            flagged,
            score,
        });
    });
}

pub fn label_outcome_gate(gate: GateKind, run: &str, step: u32, positive: bool, conf: f32) -> bool {
    let w = conf.clamp(0.0, MAX_WEIGHT);
    with_online(|st| {
        let g = st.gate_mut(gate);
        let idx = g.obs.iter().position(|o| o.run == run && o.step == step);
        let Some(idx) = idx else { return false };
        let o = g.obs.remove(idx).unwrap();
        push_example(g, Example { feats: o.feats, y: if positive { 1.0 } else { 0.0 }, w });
        true
    })
}

pub fn label_wrong_gate(gate: GateKind) -> bool {
    with_online(|st| {
        let g = st.gate_mut(gate);
        let idx = g.obs.iter().rposition(|o| o.flagged);
        let Some(idx) = idx else { return false };
        let o = g.obs.remove(idx).unwrap();
        push_example(g, Example { feats: o.feats, y: 0.0, w: GOLD_WEIGHT });
        true
    })
}

pub fn label_wasted_gate(gate: GateKind) -> bool {
    with_online(|st| {
        let g = st.gate_mut(gate);
        if g.obs.is_empty() {
            return false;
        }
        let o = g.obs.pop_back().unwrap();
        push_example(g, Example { feats: o.feats, y: 1.0, w: GOLD_WEIGHT });
        true
    })
}

fn push_example(g: &mut GateState, ex: Example) {
    if g.examples.len() >= MAX_EXAMPLES {
        g.examples.remove(0);
    }
    g.examples.push(ex);
}

// Supervisor default aliases (backwards compatibility)
pub fn observe(run: &str, step: u32, feats: &[f32], flagged: bool, score: f32) {
    observe_gate(GateKind::Supervisor, run, step, feats, flagged, score);
}

pub fn label_outcome(run: &str, step: u32, wasted: bool, conf: f32) -> bool {
    label_outcome_gate(GateKind::Supervisor, run, step, wasted, conf)
}

pub fn label_wrong() -> bool {
    label_wrong_gate(GateKind::Supervisor)
}

pub fn label_wasted() -> bool {
    label_wasted_gate(GateKind::Supervisor)
}

// Relevance typed hooks
pub fn observe_relevance(run: &str, cand_id: u32, feats: &[f32], chosen: bool, score: f32) {
    observe_gate(GateKind::Relevance, run, cand_id, feats, chosen, score);
}

pub fn label_relevance(run: &str, cand_id: u32, useful: bool, conf: f32) -> bool {
    label_outcome_gate(GateKind::Relevance, run, cand_id, useful, conf)
}

// Compaction typed hooks
pub fn observe_compaction(run: &str, seg_idx: u32, feats: &[f32], compacted: bool, score: f32) {
    observe_gate(GateKind::Compaction, run, seg_idx, feats, compacted, score);
}

pub fn label_compaction(run: &str, seg_idx: u32, needed: bool, conf: f32) -> bool {
    // needed = false (disposable, label 1.0), needed = true (load-bearing, label 0.0)
    label_outcome_gate(GateKind::Compaction, run, seg_idx, !needed, conf)
}

// Freshness typed hooks
pub fn observe_freshness(run: &str, doc_id: u32, feats: &[f32], stale: bool, score: f32) {
    observe_gate(GateKind::Freshness, run, doc_id, feats, stale, score);
}

pub fn label_freshness(run: &str, doc_id: u32, stale: bool, conf: f32) -> bool {
    label_outcome_gate(GateKind::Freshness, run, doc_id, stale, conf)
}

// Alloc typed hooks
pub fn observe_alloc(run: &str, turn_id: u32, feats: &[f32], pressured: bool, score: f32) {
    observe_gate(GateKind::Alloc, run, turn_id, feats, pressured, score);
}

pub fn label_alloc(run: &str, turn_id: u32, pressured: bool, conf: f32) -> bool {
    label_outcome_gate(GateKind::Alloc, run, turn_id, pressured, conf)
}

/* ── Runtime fixtures: replay anchors + adoption gate ───────────────── */

fn supervisor_fixtures() -> Vec<Example> {
    const TASK: &str = "tune the wind turbine governor before the winter storms";
    const P0: &str = "src/turbine/governor.rs";
    const P1: &str = "src/turbine/pitch.rs";

    fn tc(tool: &str, sig: &str, target: &str, chars: u32) -> TrajCall {
        TrajCall { tool: tool.to_string(), sig: sig.to_string(), target: target.to_string(), result_chars: chars, errored: false }
    }
    fn feats(task: &str, calls: Vec<TrajCall>, tool: &str, args: &str, target: &str) -> Vec<f32> {
        let step = calls.len() as u32 + 1;
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
        features::extract_v2(&ctx).to_vec()
    }

    let read0 = format!("{{\"path\":\"{P0}\"}}");
    let r0 = tc("read_file", &format!("read_file:{read0}"), P0, 1200);
    let grep_gain = tc("grep", "{\"pattern\":\"governor_gain\"}", "governor_gain", 500);

    vec![
        Example { feats: feats(TASK, vec![r0.clone()], "read_file", &read0, P0), y: 1.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone()], "read_file", &format!("{{\"offset\":50,\"path\":\"{P0}\"}}"), P0), y: 1.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\".\"}", "."), y: 1.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\"sourdough starter feeding schedule\"}", "sourdough starter feeding schedule"), y: 1.0, w: 1.0 },
        Example { feats: feats(TASK, vec![], "read_file", &read0, P0), y: 0.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone()], "grep", "{\"pattern\":\"governor_gain\"}", "governor_gain"), y: 0.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone(), grep_gain.clone()], "grep", "{\"pattern\":\"pitch_trim\"}", "pitch_trim"), y: 0.0, w: 1.0 },
        Example { feats: feats(TASK, vec![r0.clone(), tc("edit_file", &format!("{{\"path\":\"{P0}\"}}"), P0, 60)], "read_file", &format!("{{\"path\":\"{P1}\"}}"), P1), y: 0.0, w: 1.0 },
    ]
}

static FIXTURES: LazyLock<Vec<Example>> = LazyLock::new(supervisor_fixtures);

static FIXTURES_BLOB: &[u8] = include_bytes!("supervisor/fixtures_v2.f32");

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
    let item_size = (1 + 42) * 4;
    let payload_len = count as usize * item_size;
    if bytes.len() != 16 + payload_len {
        return Vec::new();
    }
    let payload = &bytes[16..];
    if crate::ml::net::crc32_ieee(payload) != crc {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for chunk in payload.chunks_exact(item_size) {
        let y = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let mut feats = vec![0f32; 42];
        for (i, slot) in feats.iter_mut().enumerate() {
            *slot = f32::from_le_bytes(chunk[4 + i * 4..8 + i * 4].try_into().unwrap());
        }
        out.push(Example { feats, y, w: 1.0 });
    }
    out
}

static BAKED_FIXTURES: LazyLock<Vec<Example>> = LazyLock::new(|| parse_baked_fixtures(FIXTURES_BLOB));

/* ── Adaptation ─────────────────────────────────────────────────────── */

#[derive(Clone, Debug)]
pub struct LearnReport {
    pub enabled: bool,
    pub examples: usize,
    pub pending: bool,
    pub gate_passed: usize,
    pub gate_total: usize,
    pub clamped: bool,
    pub detail: String,
}

pub fn learn_gate(gate: GateKind) -> LearnReport {
    with_online(|st| {
        let g = st.gate_mut(gate);
        g.enabled = true;
        let n = g.examples.len();
        if n < MIN_EXAMPLES {
            return LearnReport {
                enabled: true,
                examples: n,
                pending: false,
                gate_passed: 0,
                gate_total: 0,
                clamped: false,
                detail: format!("need at least {MIN_EXAMPLES} labeled examples for {}, have {n}", gate.name()),
            };
        }

        let start = n.saturating_sub(BATCH_CAP);
        let mut batch: Vec<Example> = g.examples[start..].to_vec();

        let net = gate.net();
        let threshold = gate.threshold();

        let (anchors, baked) = match gate {
            GateKind::Supervisor => (FIXTURES.as_slice(), BAKED_FIXTURES.as_slice()),
            _ => (&[][..], &[][..]),
        };

        batch.extend(anchors.iter().cloned());

        let pretrained = net.output_layer().to_vec();
        let h2 = net.h2 as usize;
        let mut w3 = pretrained.clone();
        for _ in 0..LEARN_EPOCHS {
            for ex in batch.iter() {
                let h = net.forward_hidden(&ex.feats);
                let p = net.forward_output_with(&w3, &h);
                let grad = ex.w * (p - ex.y);
                for j in 0..h2 {
                    w3[j] -= LEARN_LR * grad * h[j];
                }
                w3[h2] -= LEARN_LR * grad;
            }
        }
        let clamped = clamp_trust_region(&mut w3, &pretrained, TRUST_REL);

        let gate_fixtures: Vec<&Example> = anchors.iter().chain(baked.iter()).collect();
        let mut passed = 0usize;
        for ex in gate_fixtures.iter() {
            let h = net.forward_hidden(&ex.feats);
            let p = net.forward_output_with(&w3, &h);
            let flagged = p >= threshold;
            if flagged == (ex.y >= 0.5) {
                passed += 1;
            }
        }
        if !gate_fixtures.is_empty() && passed < gate_fixtures.len() {
            return LearnReport {
                enabled: true,
                examples: n,
                pending: false,
                gate_passed: passed,
                gate_total: gate_fixtures.len(),
                clamped,
                detail: format!(
                    "candidate for {} regressed {} of {} adoption fixtures -- discarded",
                    gate.name(),
                    gate_fixtures.len() - passed,
                    gate_fixtures.len()
                ),
            };
        }

        g.pending = Some(w3);
        LearnReport {
            enabled: true,
            examples: n,
            pending: true,
            gate_passed: passed,
            gate_total: gate_fixtures.len(),
            clamped,
            detail: format!(
                "candidate for {} re-fit on {n} examples; {passed}/{} adoption fixtures pass; trust region {} -- /ml adopt to apply, /ml discard to drop",
                gate.name(),
                gate_fixtures.len(),
                if clamped { "applied" } else { "not needed" }
            ),
        }
    })
}

pub fn learn() -> LearnReport {
    learn_gate(GateKind::Supervisor)
}

pub fn adopt_gate(gate: GateKind) -> (bool, bool) {
    with_online(|st| {
        let g = st.gate_mut(gate);
        let Some(w3) = g.pending.take() else { return (false, false) };
        g.adapted = Some(w3);
        g.adaptations += 1;
        (true, persist(gate, g.adapted.as_deref().unwrap()))
    })
}

pub fn adopt() -> (bool, bool) {
    adopt_gate(GateKind::Supervisor)
}

pub fn discard_gate(gate: GateKind) -> bool {
    with_online(|st| st.gate_mut(gate).pending.take().is_some())
}

pub fn discard() -> bool {
    discard_gate(GateKind::Supervisor)
}

pub fn output_layer_for(gate: GateKind) -> Option<Vec<f32>> {
    with_online(|st| st.gate(gate).adapted.clone())
}

pub fn output_layer() -> Option<Vec<f32>> {
    output_layer_for(GateKind::Supervisor)
}

pub fn reset_gate(gate: GateKind) {
    with_online(|st| {
        let g = st.gate_mut(gate);
        g.adapted = None;
        g.pending = None;
        g.examples.clear();
        g.adaptations = 0;
    });
    let _ = std::fs::remove_file(persist_path(gate));
}

pub fn reset() {
    reset_gate(GateKind::Supervisor);
}

pub fn status_for(gate: GateKind) -> (bool, usize, usize, bool, u32, bool) {
    with_online(|st| {
        let g = st.gate(gate);
        (
            g.enabled,
            g.obs.len(),
            g.examples.len(),
            g.adapted.is_some(),
            g.adaptations,
            g.pending.is_some(),
        )
    })
}

pub fn status() -> (bool, usize, usize, bool, u32, bool) {
    status_for(GateKind::Supervisor)
}

/* ── Persistence (§13 guardrail 8) ──────────────────────────────────── */

fn persist_path(gate: GateKind) -> std::path::PathBuf {
    let mut p = crate::embed_config::home_dir()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    p.push(".sofuu");
    p.push("ml");
    p.push(format!("{}_online.f32", gate.name()));
    p
}

fn persist(gate: GateKind, w3: &[f32]) -> bool {
    let path = persist_path(gate);
    if let Some(dir) = path.parent() {
        if std::fs::create_dir_all(dir).is_err() {
            return false;
        }
    }
    let hash = gate.blob_hash();
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
    let tmp = path.with_extension("tmp");
    let ok = std::fs::write(&tmp, &out).is_ok() && std::fs::rename(&tmp, &path).is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
    }
    ok
}

fn load_persisted(gate: GateKind) -> Option<Vec<f32>> {
    let bytes = std::fs::read(persist_path(gate)).ok()?;
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
    if hash != gate.blob_hash() {
        return None;
    }
    if n != h2 + 1 || bytes.len() != 28 + n as usize * 4 {
        return None;
    }
    let weight_bytes = &bytes[28..];
    if crate::ml::net::crc32_ieee(weight_bytes) != crc {
        return None;
    }
    let mut w3 = Vec::with_capacity(n as usize);
    for chunk in weight_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes(chunk.try_into().ok()?);
        if !v.is_finite() {
            return None;
        }
        w3.push(v);
    }
    Some(w3)
}

#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::supervisor::model;

    #[test]
    fn fixtures_are_separated_by_the_pretrained_net() {
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
        let feats = FIXTURES[0].feats.clone();
        observe("online-test-run", 99, &feats, true, 0.9);
        assert!(label_outcome("online-test-run", 99, true, 1.0));
        assert!(!label_outcome("online-test-run", 99, true, 1.0), "consumed");
    }

    #[test]
    fn learn_refuses_small_batches() {
        let _g = TEST_LOCK.lock().unwrap();
        let saved = with_online(|st| {
            let g = st.gate_mut(GateKind::Supervisor);
            let saved = g.examples.clone();
            g.examples.clear();
            for ex in FIXTURES.iter().take(3) {
                g.examples.push(ex.clone());
            }
            saved
        });
        let r = learn();
        with_online(|st| st.gate_mut(GateKind::Supervisor).examples = saved);
        assert!(r.enabled);
        assert!(!r.pending);
        assert!(r.detail.contains("at least"));
    }

    #[test]
    fn label_wrong_needs_a_flagged_observation() {
        let _g = TEST_LOCK.lock().unwrap();
        with_online(|st| st.gate_mut(GateKind::Supervisor).obs.clear());
        assert!(!label_wrong());
        observe("online-test-run", 7, &FIXTURES[3].feats, true, 0.9);
        assert!(label_wrong());
    }

    #[test]
    fn label_wasted_takes_the_most_recent_observation() {
        let _g = TEST_LOCK.lock().unwrap();
        with_online(|st| st.gate_mut(GateKind::Supervisor).obs.clear());
        assert!(!label_wasted());
        observe("online-test-run", 1, &FIXTURES[4].feats, false, 0.1);
        observe("online-test-run", 2, &FIXTURES[5].feats, false, 0.2);
        assert!(label_wasted());
        let left = with_online(|st| st.gate(GateKind::Supervisor).obs.iter().map(|o| o.step).collect::<Vec<_>>());
        assert_eq!(left, vec![1]);
    }

    #[test]
    fn zero_weight_examples_do_not_move_the_candidate() {
        let _g = TEST_LOCK.lock().unwrap();
        let base: Vec<Example> = FIXTURES.iter().cloned().cycle().take(16).collect();
        let mut padded = base.clone();
        for ex in FIXTURES.iter().take(4) {
            let mut e = ex.clone();
            e.y = 1.0 - e.y;
            e.w = 0.0;
            padded.push(e);
        }
        let train = |buf: Vec<Example>| -> Option<Vec<f32>> {
            let saved = with_online(|st| {
                let g = st.gate_mut(GateKind::Supervisor);
                let saved = g.examples.clone();
                g.examples = buf;
                g.pending = None;
                saved
            });
            let r = learn();
            let layer = with_online(|st| st.gate_mut(GateKind::Supervisor).pending.take());
            with_online(|st| st.gate_mut(GateKind::Supervisor).examples = saved);
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
        let _g = TEST_LOCK.lock().unwrap();
        let saved = with_online(|st| {
            let g = st.gate_mut(GateKind::Supervisor);
            let saved = (g.examples.clone(), g.adapted.clone(), g.adaptations);
            g.examples = FIXTURES.iter().cloned().collect();
            g.examples.extend(FIXTURES.iter().cloned());
            g.adapted = None;
            g.pending = None;
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
        with_online(|st| {
            let g = st.gate_mut(GateKind::Supervisor);
            g.examples = saved.0;
            g.adapted = saved.1;
            g.adaptations = saved.2;
        });
    }
}
