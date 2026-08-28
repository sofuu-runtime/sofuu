// ml/alloc — Model 5: the context-window allocator.
//
// Allocates input/output/window config RESPECTING THE SELECTED MODEL:
// the mechanical layer (policy.rs) resolves the model's real limits
// first (caps registry → provider-error-learned → conservative
// defaults), clamps explicit config to them, and parses real limits out
// of provider limit errors so the same mistake is never made twice. The
// learned layer (model.rs) predicts context pressure and turns it into
// the per-turn allocation — when to compact, how much window tools /
// recall / attachments may take, how much output to reserve — all
// clamped inside the resolved limits. The pre-flight fit check at the
// request-build sites (agent.js / chat driver / rlm.js) applies the plan
// and corrects BEFORE sending, so an oversized request never reaches
// the provider. Advise-only (Principle 1): the caller acts, the
// drop-oldest guard and compaction cliff stay the final nets.

pub mod features;
pub mod model;
pub mod policy;

#[cfg(test)]
mod eval;
