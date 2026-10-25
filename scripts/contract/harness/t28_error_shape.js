// t28_error_shape.js — R2.5: one error shape and one taxonomy on every call.
//
// Run: ./sofuu run scripts/contract/harness/t28_error_shape.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG"); process.exit(1); }, 15000);

const P = (s) => JSON.parse(s);

function main() {
    // The §1.4 shape: code, op, surface, retryable, message (+optionals).
    const e = P(sofuu.async.makeError(JSON.stringify({
        code: "E_TIMEOUT", op: "ai.stream", surface: "http",
        deadline_ms: 120000, elapsed_ms: 120003, retryable: true,
        message: "no first token before the deadline", turn_id: "t1", seq: 41,
    })));
    assert("code carried", e.code === "E_TIMEOUT");
    assert("op carried", e.op === "ai.stream");
    assert("surface carried", e.surface === "http");
    assert("retryable carried", e.retryable === true);
    assert("deadline_ms/elapsed_ms carried", e.deadline_ms === 120000 && e.elapsed_ms === 120003);
    assert("turn_id/seq carried", e.turn_id === "t1" && e.seq === 41);

    // Retryability defaults come from the taxonomy, not the caller.
    const cap = P(sofuu.async.makeError(JSON.stringify({ code: "E_CAPABILITY", op: "memory.remember" })));
    assert("E_CAPABILITY defaults to non-retryable", cap.retryable === false);
    const net = P(sofuu.async.makeError(JSON.stringify({ code: "E_NETWORK", op: "fetch" })));
    assert("E_NETWORK defaults to retryable", net.retryable === true);

    // Unknown codes wrap as E_INTERNAL — no taxonomy invention at the edges.
    const w = P(sofuu.async.makeError(JSON.stringify({ code: "E_WAT", op: "x" })));
    assert("unknown code wraps as E_INTERNAL", w.code === "E_INTERNAL" && String(w.note).indexOf("E_WAT") >= 0);

    // The full taxonomy is enumerable and matches the spec's ten codes.
    const codes = P(sofuu.async.codes()).codes;
    const want = ["E_CAPABILITY", "E_TIMEOUT", "E_CANCELLED", "E_BACKPRESSURE", "E_RATE_LIMIT",
        "E_AUTH", "E_NETWORK", "E_PROTOCOL", "E_SCHEMA", "E_INTERNAL"];
    assert("taxonomy has exactly the ten spec codes",
        codes.length === 10 && want.every((c) => codes.some((x) => x.code === c)));

    console.log(`\n=== t28 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
