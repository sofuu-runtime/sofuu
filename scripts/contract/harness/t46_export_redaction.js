// t46_export_redaction.js — R4.6: an export of a session that contains
// secrets contains none — redaction is part of the format, on both the
// append path (R6.6) and the document import/export path.
//
// Run: ./sofuu run scripts/contract/harness/t46_export_redaction.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 20 s"); process.exit(1); }, 20000);

const P = (s) => JSON.parse(s);

async function main() {
    const SID = "s-3f8c1a02d9e7b-t4" + (860000000 + (Date.now() % 100000000));

    // A crafted document carrying secrets in the places an app might leave
    // them: the header, a turn's note, and a tool record.
    const doc = {
        schema_version: 1,
        session: { id: SID, created: "2026-10-26T09:00:00Z", updated: "2026-10-26T09:01:00Z",
                   project: ".", title: "leaky", model: "m", provider: "openai",
                   api_key: "sk-live-abcdef123456", Authorization: "Bearer zz.top" },
        turns: [{
            id: SID + "-t1", index: 0, prompt: "hi", answer: "hello",
            status: "done",
            usage: { in: 3, out: 5, model: "m", provider: "openai", cost_usd: 0 },
            tools: [{ id: "c1", name: "bash", ok: true, ms: 1,
                      env_token: "ghp_supplyplz" }],
            served_by: "cloud", error: null,
        }],
        events: { format: "ndjson", path: "events.ndjson" },
    };

    const warned = console.log;
    console.log = () => {}; // import prints its report; drop from the fixture output
    const im = P(sofuu.session.import(JSON.stringify(doc)));
    console.log = warned;
    assert("import accepts the well-formed doc", im.ok === true && im.id === SID);

    const out = P(sofuu.session.export(JSON.stringify({ project: ".", sid: SID })));
    assert("export drops session.api_key", out.session.api_key === undefined || out.session.api_key === "[REDACTED]");
    assert("export drops session.Authorization",
        out.session.Authorization === undefined || out.session.Authorization === "[REDACTED]");
    const blob = JSON.stringify(out);
    assert("no raw secret bytes anywhere in the export", blob.indexOf("sk-live-abcdef123456") < 0
        && blob.indexOf("Bearer zz.top") < 0 && blob.indexOf("ghp_supplyplz") < 0);
    assert("the turn data itself survived redaction (lossless fields)",
        out.turns[0].answer === "hello" && out.turns[0].tools[0].id === "c1"
        && out.turns[0].usage.in === 3);
    assert("usage still canonical after the round-trip",
        out.turns[0].usage.in === 3 && out.turns[0].usage.out === 5
        && out.turns[0].usage.model === "m");

    console.log(`\n=== t46 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t46:", (e && e.message) || e); process.exit(1); });
