// recall_eval — Cargo shim for the W4.5 harness.
//
// QTSQ_UPGRADE_PLAN.md names the harness `scripts/stress/recall_eval.rs`, but
// Cargo only builds example targets that live in the package's examples/
// directory. This file is that target: it `include!`s the plan-named source so
// there is ONE implementation, at the path the plan references.
//
//   cargo run --release -p sofuu-core --example recall_eval -- --n 2000 --dim 128 --ef 64
//
// Edit scripts/stress/recall_eval.rs, not this shim.

include!("../../../scripts/stress/recall_eval.rs");
