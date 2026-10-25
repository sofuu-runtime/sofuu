// t62_eventlog.js — R6.2/R2.4: the session event log is the turn's source of
// truth, with a total order and bounded queues.
//
// Run: ./sofuu run scripts/contract/harness/t62_eventlog.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG"); process.exit(1); }, 20000);

const P = (s) => JSON.parse(s);

function ev(kind, turn, seq, payload) {
    return {
        kind, schema_version: 1, sid: "s_harness", turn_id: turn, seq,
        ts: "2026-10-25T12:00:00Z", surface: "js", payload,
    };
}

function main() {
    const project = "/tmp/sofuu-t62-" + (process.pid || 1);
    const arg = (events) => JSON.stringify({ project, sid: "s_harness", events });

    // ── 1. append a turn, read it back ─────────────────────────────────
    let res = P(sofuu.events.append(arg([
        ev("delta", "t1", 1, { text: "Hel" }),
        ev("delta", "t1", 2, { text: "lo" }),
        ev("tool", "t1", 3, { id: "c1", name: "grep" }),
        ev("tool_result", "t1", 4, { id: "c1", name: "grep", ok: true, ms: 5 }),
        ev("usage", "t1", 5, { in: 3, out: 2, model: "m", provider: "local" }),
        ev("stop", "t1", 6, { reason: "done" }),
    ])));
    assert("append validates and writes all 6 events", res.ok === true && res.written === 6);

    const read = P(sofuu.events.read(JSON.stringify({ project, sid: "s_harness" })));
    assert("read returns 6 valid events", read.ok === true && read.count === 6);

    // ── 2. R6.2: the log alone reconstructs the turn ───────────────────
    const rec = P(sofuu.events.reconstruct(JSON.stringify({ project, sid: "s_harness" })));
    const t = (rec.turns || [])[0] || {};
    assert("reconstructed answer from deltas", t.answer === "Hello");
    assert("reconstructed tool pairing", t.tools && t.tools[0] && t.tools[0].name === "grep" && t.tools[0].ok === true);
    assert("reconstructed usage totals", t.usage && t.usage.out === 2);
    assert("reconstructed terminal state", t.stop_reason === "done");

    // ── 3. R6.3: a seq regression is rejected, never written ───────────
    res = P(sofuu.events.append(arg([ev("delta", "t1", 2, { text: "late" })])));
    assert("seq regression rejected", res.ok === false && res.written === 0);

    // ── 4. R2.4: bounded queues overflow with a typed error ────────────
    const flood = [];
    for (let i = 0; i < 1100; i++) flood.push(ev("delta", "t2", 100 + i, { text: "x" }));
    res = P(sofuu.events.append(arg(flood)));
    assert("queue overflow reports E_BACKPRESSURE", res.code === "E_BACKPRESSURE");

    // ── 5. cleanup ──────────────────────────────────────────────────────
    try { sofuu.fs.rm(project, { recursive: true }); } catch (e) {
        try { sofuu.fs.rm(project + "/.sofuu/sessions/s_harness/events.ndjson"); } catch (e2) {}
    }

    console.log(`\n=== t62 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
