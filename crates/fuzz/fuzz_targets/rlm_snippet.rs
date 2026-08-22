//! cargo-fuzz target for the RLM sandbox (sofuu_core::rlm::sandbox).
//!
//! Arbitrary bytes as a model-written JS snippet into a small-context
//! sandbox: eval must never panic or abort the host — Outcome::Error
//! (syntax/runtime error, deadline interrupt) is a fine result.
//!
//! Harness note: one sandbox is reused per process (creating a QuickJS
//! runtime per input would dominate the run and is not what we're
//! testing). The deadline is re-armed per eval (the per-eval slice added
//! for the RLM security batch) so a single timed-out eval doesn't expire
//! the whole fuzz run; after ANY error the sandbox is recreated, because
//! an interrupted/OOM'd runtime is documented as possibly poisoned.

#![no_main]

use std::cell::RefCell;
use std::time::Duration;

use libfuzzer_sys::fuzz_target;
use sofuu_core::rlm::sandbox::{Outcome, Sandbox};

thread_local! {
    static SB: RefCell<Option<Sandbox>> = const { RefCell::new(None) };
}

fuzz_target!(|data: &[u8]| {
    let src = String::from_utf8_lossy(data);
    SB.with(|cell| {
        let mut g = cell.borrow_mut();
        let fresh = || {
            let mut s = Sandbox::with_defaults(Duration::from_secs(60)).expect("sandbox");
            s.set_context("fuzz context alpha beta gamma", 64);
            s
        };
        if g.is_none() {
            *g = Some(fresh());
        }
        let s = g.as_mut().expect("sandbox present");
        // Short per-eval slice: a `while(1){}` input costs 200ms, not 60s.
        s.reset_eval_deadline(Duration::from_millis(200));
        let outcome = s.eval_snippet(&src, &[]);
        if matches!(outcome, Outcome::Error { .. }) {
            // Possibly poisoned (interrupt/OOM) — don't reuse it.
            *g = Some(fresh());
        }
    });
});
