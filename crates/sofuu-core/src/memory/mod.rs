// sofuu-core — memory subsystem (CMA cognitive memory).
pub mod cma;
pub mod hnsw;
/// B5.4: the recall evaluation harness (tests only — it exists to measure
/// ranking changes, not to ship).
#[cfg(test)]
pub mod recall_eval;
/// Brain Core II M0: the model-based invariant fuzz suite (the precision
/// floor every brain change is gated on).
#[cfg(test)]
pub mod invariant_test;
