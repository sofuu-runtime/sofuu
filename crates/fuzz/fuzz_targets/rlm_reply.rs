//! cargo-fuzz target for the RLM episode loop (sofuu_core::rlm::episode).
//!
//! Arbitrary bytes as a model reply into Episode::step — snippet
//! extraction, error paths, and budget accounting must never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sofuu_core::rlm::episode::Episode;
use sofuu_core::rlm::{RlmOpts, RlmRequest};

fuzz_target!(|data: &[u8]| {
    let reply = String::from_utf8_lossy(data);
    let req = RlmRequest {
        context: "fuzz context: alpha beta gamma ".repeat(8),
        question: "q".to_string(),
        opts: RlmOpts::default(),
    };
    // One fresh episode per input — the interesting surface is the reply,
    // and episode construction is cheap next to the sandbox it owns.
    if let Ok(mut ep) = Episode::new(req, 0) {
        let _ = ep.start();
        let _ = ep.step(&reply); // must never panic on arbitrary bytes
        let _ = ep.step(""); // a second, empty reply: degrade, don't panic
        // Protocol misuse with nothing suspended: transcript resend.
        let _ = ep.feed_llm_answers(vec!["x".to_string()]);
    }
});
