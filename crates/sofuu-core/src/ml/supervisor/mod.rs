// ml/supervisor — the real-time supervisor gate (PLAN-ML-GATES §11).
//
// Phase 5. Answers "is this action the right one right now?" at the
// decision points — before a tool runs (the only place waste can be
// PREVENTED) and at each loop boundary (trajectory-level: spinning,
// drifting, over budget). Advise-only (Principle 1): the nudge rides the
// tool result or the ephemeral context message back in-band; the call
// still runs, the LLM decides. features.rs is the 33-feature trajectory/
// action extractor; model.rs the trained net + policy layered over the
// Phase-1 mechanical rules (context.rs dup_call / reread_unchanged);
// eval.rs the committed-weights fixtures.

pub mod features;
pub mod model;

#[cfg(test)]
mod eval;
