// ml/compaction — Model 4: when and what to compact? (PLAN-ML-GATES §12)
//
// Replaces the COMPACT_AT cliff with scored, incremental passes: the net
// scores each history segment (disposable vs load-bearing), the policy
// layer frees a SMALL budget per pass (~10-15% of the window), prefers
// the free tier (dedupe / boilerplate drop / truncation — no LLM), and
// the drop-oldest guard stays the final safety net. The net SELECTS; it
// never writes prose — summarization of flagged segments still uses the
// LLM, but scoped and incremental.

pub mod features;
pub mod model;

#[cfg(test)]
mod eval;
