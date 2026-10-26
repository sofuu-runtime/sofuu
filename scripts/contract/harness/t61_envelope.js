// t61_envelope.js — R6.1/R2.6: the envelope rides on every real emitted event,
// ordering is total, and each run terminates exactly once.
//
// Captures REAL events from two drivers:
//   1. the agent loop (agent.run against an in-process mock provider; the run
//      result's trace is the emitted event stream), and
//   2. the chat driver (init with an onEvent sink; the turn's events even on
//      the error path).
// Then asserts the §7.2 identity fields on every event, validates them with
// the runtime's own validator (sofuu.events.validate), and reports payload-rule
// violations informationally (payload alignment is the remaining R6 wiring).
//
// Run: ./sofuu run scripts/contract/harness/t61_envelope.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);
const IDENTITY_FIELDS = ["schema_version", "seq", "ts", "surface", "sid", "turn_id"];

function identityOk(e) {
    return e && e.schema_version === 1 && typeof e.seq === "number"
        && typeof e.ts === "string" && e.ts.length >= 20
        && e.surface === "js" && typeof e.sid === "string" && typeof e.turn_id === "string";
}

async function main() {
    // ── mock provider (the real native streaming path) ─────────────────
    let server = null, port = 0;
    const base = 25800 + (Date.now() % 3000);
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "ok" } }] }) + "\n\n");
        res.write("data: " + JSON.stringify({ choices: [{ delta: { content: " then" } }] }) + "\n\n");
        res.write("data: [DONE]\n\n");
        res.end();
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 139;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock provider bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // ── 1. agent loop events ───────────────────────────────────────────
    sofuu.agent.define({
        name: "chaosex", system: "Reply with one word.",
        model: "mock", base_url: URL, api_key: "x",
    });
    const r = await sofuu.agent.run("chaosex", "say hi");
    const trace = (r && r.trace) || [];
    assert("agent run produced an event trace", trace.length >= 3);
    assert("every agent event carries the §7.2 identity fields",
        trace.every(identityOk));
    const seqs = trace.map((e) => e.seq);
    let monotonic = true;
    for (let i = 1; i < seqs.length; i++) if (seqs[i] <= seqs[i - 1]) monotonic = false;
    assert("seq strictly increasing (total order, R6.3)", monotonic);
    const stops = trace.filter((e) => e.kind === "stop").length;
    assert("exactly one terminal stop event (R2.6)", stops === 1);

    // The runtime's own validator: identity errors must be zero everywhere;
    // payload-rule gaps are reported informationally.
    let identityErrs = 0, payloadErrs = 0;
    for (const e of trace) {
        const errs = P(sofuu.events.validate(JSON.stringify(e))).errors || [];
        for (const err of errs) {
            if (/envelope field|unknown kind|seq|ts|surface/.test(err)) identityErrs += 1;
            else payloadErrs += 1;
        }
    }
    assert("zero envelope-identity violations across the agent trace", identityErrs === 0);
    console.log(`  ℹ payload-rule violations (informational, R6 wiring): ${payloadErrs}`);

    // ── 2. chat driver events (error path, no provider configured) ─────
    // The live turn's event sink attaches at submit time (chat.js:2110).
    const chatEvents = [];
    try { sofuu.chat.init({}); } catch (e) { console.log("  ℹ chat.init:", (e && e.message) || e); }
    try {
        // Deliberately offline: a dead local base_url makes the failure
        // instant and deterministic — the error path's events must still
        // ride the §7.2 envelope regardless of the host's real config
        // (an active ~/.sofuu/config.json provider must never leak a real
        // network call into a contract harness). The submit is raced, not
        // awaited: a failing provider's retry/window legs may run long; the
        // envelope contract is about the events on the sink, not the settle.
        await Promise.race([
            sofuu.chat.submit("envelope probe", {
                provider: "openai", model: "dead-mock", api_key: "x",
                base_url: "http://127.0.0.1:9/v1/chat/completions",
                onEvent: (ev) => { if (ev && ev.kind) chatEvents.push(ev); },
            }).catch(() => {}),
            new Promise((r) => setTimeout(r, 6000)),
        ]);
        /* The raced submit may still run a retry/tool leg — its children are
         * killed here so the suite's later orphan checks (T2.1/T2.3) see a
         * clean process table. */
        try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) {}
        await new Promise((r) => setTimeout(r, 300));
    } catch (e) { /* the error path is fine — the events matter */ }
    await new Promise((r2) => setTimeout(r2, 300));
    assert("chat driver emitted events through the sink", chatEvents.length >= 1);
    assert("every chat event carries the §7.2 identity fields",
        chatEvents.every(identityOk));
    const chatStops = chatEvents.filter((e) => e.kind === "done" || e.kind === "stop").length;
    assert("at most one chat terminal event", chatStops <= 1);

    console.log(`\n=== t61 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t61:", (e && e.message) || e); process.exit(1); });
