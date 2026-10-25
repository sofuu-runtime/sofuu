// t66_redaction.js — R6.6: no secret reaches a log, trace, or export.
//
// Run: ./sofuu run scripts/contract/harness/t66_redaction.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG"); process.exit(1); }, 15000);

function main() {
    const event = {
        kind: "tool", schema_version: 1, sid: "s1", turn_id: "t1", seq: 1,
        ts: "2026-10-25T12:00:00Z", surface: "js",
        payload: {
            id: "c1", name: "net",
            api_key: "sk-live-1234567890",
            Authorization: "Bearer abc.def.ghi",
            nested: { session_token: "supersecret", note: "keep me" },
            safe: "plain value",
        },
    };

    const out = JSON.parse(sofuu.events.redact(JSON.stringify(event)));
    const s = JSON.stringify(out);
    assert("api_key redacted", !s.includes("sk-live-1234567890"));
    assert("Authorization redacted", !s.includes("Bearer abc"));
    assert("nested token redacted", !s.includes("supersecret"));
    assert("redaction marker present", s.includes("[REDACTED]"));
    assert("non-secret values preserved", s.includes("keep me") && s.includes("plain value"));
    assert("envelope survives redaction",
        out.kind === "tool" && out.schema_version === 1 && out.seq === 1);

    // The redacted event still validates (payload rules intact).
    const v = JSON.parse(sofuu.events.validate(JSON.stringify(out)));
    assert("redacted event still validates", v.ok === true);

    console.log(`\n=== t66 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
