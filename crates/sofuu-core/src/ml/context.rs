// ml/context.rs — the shared in-memory context working set (PLAN-ML-GATES §8).
//
// A structured representation of what the agent loop currently holds — the
// per-run tool trajectory (calls, targets, outcomes) and history segments
// (type/size/age) — that the four tiny models (§3) reason over. Today the
// history is a flat {role, content} array with none of that metadata; this
// is the metadata layer.
//
// The one rule that keeps it safe: RAM is the hot layer, disk stays the
// source of truth. The .qtsq session files (session.rs) remain the durable
// store; this set is volatile and rebuilt from disk on resume. A crash loses
// nothing that was not already persisted.
//
// Phase 1 consumes the trajectory: rule-based repeat-call / re-read
// detection (the no-model subset of PLAN-ML-GATES §11). The compaction
// (phase 3) and supervisor (phase 5) models read the same structs.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Bounds: the set must stay capped even on pathological runs (sub-agent
/// fan-out creates one RunState per runId; a chat session runs for hours).
const MAX_RUNS: usize = 64;
const MAX_CALLS: usize = 256;
const MAX_READS_PER_PATH: usize = 16;
const MAX_SEGMENTS: usize = 1024;

#[derive(Clone, Debug)]
pub struct CallRec {
    pub step: u32,
    pub tool: String,
    /// Canonical call signature (tool + sorted-key args JSON, computed
    /// JS-side) — identity key for exact-repeat detection.
    pub sig: String,
    /// Primary target of the call (path/pattern), "" when none.
    pub target: String,
    pub result_chars: u32,
    pub errored: bool,
    /// postcall arrived — the result is accounted for.
    pub done: bool,
}

/// One history segment — the unit compaction (phase 3) selects over.
#[derive(Clone, Debug)]
pub struct Segment {
    pub id: u32,
    /// 0 instruction · 1 answer · 2 tool_call · 3 tool_result
    pub kind: u8,
    pub chars: u32,
    pub tokens: u32,
    pub tool: String,
    pub target: String,
    pub compacted: bool,
}

#[derive(Clone, Debug)]
pub struct RunState {
    pub run_id: String,
    pub task: String,
    pub calls: Vec<CallRec>,
    /// Canonical signature → step of the FIRST call with it (dup detection).
    pub sig_first: HashMap<String, u32>,
    /// File path → steps it was read at (re-read detection).
    pub reads: HashMap<String, Vec<u32>>,
    /// File path → last step it was written/edited (invalidates re-reads).
    pub writes: HashMap<String, u32>,
    pub segments: Vec<Segment>,
    pub next_seg_id: u32,
    pub finished: bool,
}

impl RunState {
    fn new(run_id: &str, task: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            task: task.to_string(),
            calls: Vec::new(),
            sig_first: HashMap::new(),
            reads: HashMap::new(),
            writes: HashMap::new(),
            segments: Vec::new(),
            next_seg_id: 1,
            finished: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WorkingSet {
    runs: HashMap<String, RunState>,
    /// Insertion order for LRU-style eviction (oldest first).
    order: Vec<String>,
}

/// The verdict of a pre-call rule check. `ok` = nothing to flag; otherwise
/// `reason` is the machine label and `nudge` the in-band advisory text.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub ok: bool,
    pub reason: String,
    pub nudge: Option<String>,
}

static WORKSET: LazyLock<Mutex<WorkingSet>> = LazyLock::new(|| {
    Mutex::new(WorkingSet {
        runs: HashMap::new(),
        order: Vec::new(),
    })
});

fn with_ws<R>(f: impl FnOnce(&mut WorkingSet) -> R) -> R {
    let mut ws = WORKSET.lock().unwrap();
    f(&mut ws)
}

/// Create (or reset) the state for one run. Sub-agents get their own runId,
/// so nested runs never pollute the parent's trajectory.
pub fn run_start(run_id: &str, task: &str) {
    with_ws(|ws| {
        if !ws.runs.contains_key(run_id) {
            ws.order.push(run_id.to_string());
            while ws.order.len() > MAX_RUNS {
                let oldest = ws.order.remove(0);
                ws.runs.remove(&oldest);
            }
        }
        ws.runs.insert(run_id.to_string(), RunState::new(run_id, task));
    });
}

pub fn run_end(run_id: &str) {
    with_ws(|ws| {
        if let Some(r) = ws.runs.get_mut(run_id) {
            r.finished = true;
        }
    });
}

/// Pre-call rule check (PLAN-ML-GATES §11 rule subset). Detects the two most
/// common token burns with no model at all:
///   1. `dup_call` — an identical call (same tool, same canonical args) was
///      already made this run;
///   2. `reread_unchanged` — read_file on a path already read this run with
///      no write/edit to it since.
/// Advises only: the call is still recorded and still runs; the nudge rides
/// the tool result back into the message array.
pub fn precheck(run_id: &str, step: u32, tool: &str, sig: &str, target: &str) -> Verdict {
    with_ws(|ws| {
        let r = ws
            .runs
            .entry(run_id.to_string())
            .or_insert_with(|| RunState::new(run_id, ""));

        let mut v = Verdict {
            ok: true,
            ..Default::default()
        };

        // Rule 1: exact repeat of an earlier call.
        if !sig.is_empty() {
            if let Some(&first) = r.sig_first.get(sig) {
                if first < step {
                    v.ok = false;
                    v.reason = "dup_call".to_string();
                    v.nudge = Some(format!(
                        "identical call already made at step {} (same tool, same arguments) — reuse that result unless you expect different output",
                        first
                    ));
                }
            } else {
                r.sig_first.insert(sig.to_string(), step);
            }
        }

        // Rule 2: re-reading a file that has not changed since the last read.
        if v.ok && tool == "read_file" && !target.is_empty() {
            if let Some(steps) = r.reads.get(target) {
                if let Some(&last) = steps.last() {
                    let written_since = r
                        .writes
                        .get(target)
                        .map(|&w| w > last)
                        .unwrap_or(false);
                    if !written_since {
                        v.ok = false;
                        v.reason = "reread_unchanged".to_string();
                        v.nudge = Some(format!(
                            "this file was already read at step {} and has not been modified since — reuse the earlier content",
                            last
                        ));
                    }
                }
            }
        }

        // Record the call regardless of the verdict (advisors never block).
        if !sig.is_empty() && !r.sig_first.contains_key(sig) {
            r.sig_first.insert(sig.to_string(), step);
        }
        if tool == "read_file" && !target.is_empty() {
            let steps = r.reads.entry(target.to_string()).or_default();
            steps.push(step);
            if steps.len() > MAX_READS_PER_PATH {
                steps.remove(0);
            }
        }
        if r.calls.len() >= MAX_CALLS {
            r.calls.remove(0);
        }
        r.calls.push(CallRec {
            step,
            tool: tool.to_string(),
            sig: sig.to_string(),
            target: target.to_string(),
            result_chars: 0,
            errored: false,
            done: false,
        });

        v
    })
}

/// Post-call accounting: result size + outcome, and write/edit tracking that
/// invalidates the re-read rule for later reads of the same path.
pub fn postcall(run_id: &str, step: u32, tool: &str, target: &str, chars: u32, errored: bool) {
    with_ws(|ws| {
        let Some(r) = ws.runs.get_mut(run_id) else {
            return;
        };
        if tool == "write_file" || tool == "edit_file" {
            if !target.is_empty() {
                r.writes.insert(target.to_string(), step);
            }
        }
        if let Some(c) = r.calls.iter_mut().rev().find(|c| c.step == step && !c.done) {
            c.result_chars = chars;
            c.errored = errored;
            c.done = true;
        }
    });
}

/// Record one history segment (consumer: the compaction model, phase 3 —
/// recorded from day one so the trajectory is complete when it lands).
pub fn track_segment(run_id: &str, kind: u8, chars: u32, tokens: u32, tool: &str, target: &str) {
    with_ws(|ws| {
        let r = ws
            .runs
            .entry(run_id.to_string())
            .or_insert_with(|| RunState::new(run_id, ""));
        if r.segments.len() >= MAX_SEGMENTS {
            r.segments.remove(0);
        }
        let id = r.next_seg_id;
        r.next_seg_id += 1;
        r.segments.push(Segment {
            id,
            kind,
            chars,
            tokens,
            tool: tool.to_string(),
            target: target.to_string(),
            compacted: false,
        });
    });
}

/// Aggregate counters for `sofuu.ml.workset()` / `/ml info`.
pub fn summary() -> (usize, usize, usize, u64) {
    with_ws(|ws| {
        let runs = ws.runs.len();
        let mut calls = 0usize;
        let mut segs = 0usize;
        let mut seg_tokens = 0u64;
        for r in ws.runs.values() {
            calls += r.calls.len();
            segs += r.segments.len();
            seg_tokens += r.segments.iter().map(|s| s.tokens as u64).sum::<u64>();
        }
        (runs, calls, segs, seg_tokens)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dup_call_detected() {
        run_start("t-dup", "task");
        let v1 = precheck("t-dup", 1, "grep", "grep:{\"pattern\":\"foo\"}", "");
        assert!(v1.ok, "first call is clean");
        let v2 = precheck("t-dup", 2, "grep", "grep:{\"pattern\":\"foo\"}", "");
        assert!(!v2.ok);
        assert_eq!(v2.reason, "dup_call");
        assert!(v2.nudge.unwrap().contains("step 1"));
        // Different args are not a duplicate.
        let v3 = precheck("t-dup", 3, "grep", "grep:{\"pattern\":\"bar\"}", "");
        assert!(v3.ok);
    }

    #[test]
    fn reread_unchanged_detected() {
        run_start("t-reread", "task");
        let v1 = precheck("t-reread", 1, "read_file", "read_file:{\"path\":\"a.rs\"}", "a.rs");
        assert!(v1.ok);
        let v2 = precheck("t-reread", 2, "read_file", "read_file:{\"path\":\"a.rs\"}", "a.rs");
        // Same sig fires rule 1 first — the duplicate is the stronger signal.
        assert_eq!(v2.reason, "dup_call");
        // Same file, different args (e.g. an offset read) → rule 2.
        let v3 = precheck(
            "t-reread",
            3,
            "read_file",
            "read_file:{\"path\":\"a.rs\",\"offset\":10}",
            "a.rs",
        );
        assert!(!v3.ok);
        assert_eq!(v3.reason, "reread_unchanged");
    }

    #[test]
    fn write_invalidates_reread() {
        run_start("t-write", "task");
        precheck("t-write", 1, "read_file", "read_file:{\"path\":\"b.rs\"}", "b.rs");
        postcall("t-write", 2, "edit_file", "b.rs", 100, false);
        let v = precheck(
            "t-write",
            3,
            "read_file",
            "read_file:{\"path\":\"b.rs\",\"offset\":1}",
            "b.rs",
        );
        assert!(v.ok, "a write after the last read makes re-reading legitimate");
    }

    #[test]
    fn runs_evicted_beyond_cap() {
        for i in 0..(MAX_RUNS + 8) {
            run_start(&format!("t-evict-{i}"), "");
        }
        let (runs, _, _, _) = summary();
        assert!(runs <= MAX_RUNS, "working set stays bounded");
    }

    #[test]
    fn segments_recorded_and_capped() {
        run_start("t-seg", "task");
        for i in 0..20 {
            track_segment("t-seg", 3, 100, 25, "read_file", "x.rs");
            assert!(i < 10000);
        }
        let (_, _, segs, tokens) = summary();
        assert!(segs >= 20);
        assert!(tokens >= 500);
    }
}
