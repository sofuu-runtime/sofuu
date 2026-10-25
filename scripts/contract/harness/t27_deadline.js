// t27_deadline.js — R2.2: nested deadline inheritance.
//
// Run: ./sofuu run scripts/contract/harness/t27_deadline.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG"); process.exit(1); }, 15000);

const P = (s) => JSON.parse(s);

function main() {
    // Child within the parent budget: untouched.
    let r = P(sofuu.async.childDeadline(JSON.stringify({ parent_ms: 10000, child_ms: 5000 })));
    assert("child within budget is untouched (5000)", r.ok === true && r.deadline_ms === 5000 && r.inherited === false);

    // Child beyond the budget: clamped to parent minus the margin.
    r = P(sofuu.async.childDeadline(JSON.stringify({ parent_ms: 10000, child_ms: 60000 })));
    assert("child clamped to parent minus margin (9000)", r.ok === true && r.deadline_ms === 9000 && r.inherited === true);

    // Custom margin.
    r = P(sofuu.async.childDeadline(JSON.stringify({ parent_ms: 10000, child_ms: 60000, margin_ms: 2000 })));
    assert("custom margin honoured (8000)", r.ok === true && r.deadline_ms === 8000);

    // Exhausted parent: typed timeout, never a negative deadline.
    r = P(sofuu.async.childDeadline(JSON.stringify({ parent_ms: 500, child_ms: 60000 })));
    assert("exhausted parent reports E_TIMEOUT", r.code === "E_TIMEOUT");

    // Limits carry the per-operation defaults (SPEC §3.2).
    const limits = P(sofuu.async.limits());
    assert("bounded-queue limit present (1024 events)", limits.max_queue_events === 1024);
    assert("default timeouts present", limits.default_timeouts.tool_ms === 60000 && limits.default_timeouts.ttft_ms === 30000);

    console.log(`\n=== t27 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
