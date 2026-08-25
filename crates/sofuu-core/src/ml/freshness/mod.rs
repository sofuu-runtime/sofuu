// ml/freshness — Model 1: is this material current? (PLAN-ML-GATES §5)
//
// The easiest of the four gates: dates and version strings are extractable,
// staleness vocabulary is real signal, and the answer doesn't depend on who
// is reading. Arch 28 → 112 → 48 → 1 = 8,721 params.

pub mod features;
pub mod model;

#[cfg(test)]
mod eval;
