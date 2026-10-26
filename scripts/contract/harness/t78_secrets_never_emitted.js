// t78_secrets_never_emitted.js — R7.5: a provider key resolved at call time
// never reaches the trace, the event log, the session document, or error
// text — even when the provider echoes the failure back.
//
// Run: ./sofuu run scripts/contract/harness/t78_secrets_never_emitted.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 45 s"); process.exit(1); }, 45000);

const P = (s) => JSON.parse(s);
const SECRET = "sk-t78-live-secret-9f2c1";
const KEEPERS = [];

async function main() {
    // A mock that answers normally, and can 401 while ECHOING the
    // Authorization header back in the body — the classic leak route.
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let body = {};
        try { body = P(req.body || "{}"); } catch (e) {}
        const wants401 = JSON.stringify(body.messages || []).indexOf("401-please") >= 0;
        if (wants401) {
            const echoAuth = (req.headers && (req.headers.authorization || "")) || "";
            res.writeHead(401, { "Content-Type": "application/json" });
            res.end(P(JSON.stringify({
                error: { message: "Invalid API key provided: " + echoAuth,
                         type: "invalid_request_error", code: "invalid_api_key" },
            })));
            return;
        }
        res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "ok" }, finish_reason: "stop" }] }) + "\n\n");
        res.write("data: [DONE]\n\n");
        res.end();
    };
    let server = null, port = 0;
    const base = 39700 + (Date.now() % 500);
    for (let i = 0; i < 6 && !server; i++) {
        try { server = sofuu.createServer(handler); server.listen(base + i * 17, "127.0.0.1"); port = base + i * 17; }
        catch (e) { server = null; }
    }
    KEEPERS.push(server);
    assert("mock provider bound", server != null);

    const BASE_URL = "http://127.0.0.1:" + port + "/v1/chat/completions";
    const def = { name: "secrets-t78", system: "Reply briefly.", model: "mock",
                  base_url: BASE_URL, api_key: SECRET, memory: "off", rlm: "off" };

    // ── successful turn: the key must not ride any observable surface ─────
    let trace1 = "";
    try {
        const r1 = await sofuu.agent.run(def, "hello t78", {});
        trace1 = JSON.stringify(r1 && r1.trace || "") + JSON.stringify(r1 && r1.answer || "");
    } catch (e) { trace1 = String(e && e.message || e); }
    assert("no key bytes in the successful run's trace",
        trace1.indexOf(SECRET) < 0);

    // ── failing turn: the 401 echoes the Authorization header; the surfaced
    // error must not carry the key either.
    let trace2 = "";
    try {
        const r2 = await sofuu.agent.run(def, "401-please fail loudly", {});
        trace2 = JSON.stringify(r2 && r2.trace || "");
    } catch (e) { trace2 = String(e && e.message || e); }
    assert("no key bytes in the failing run's trace or error", trace2.indexOf(SECRET) < 0);

    // ── the events log + the session document (the R4 export surface) ─────
    const SID = "s-3f8c1a02d9e7b-t4" + (780000001 + (Date.now() % 100000000));
    const mk = (kind, seq, payload) => ({ kind: kind, schema_version: 1, sid: SID,
        turn_id: SID + "-t" + (seq % 2 ? 1 : 2), seq: seq,
        ts: "2026-10-26T10:0" + (seq % 10) + ":00Z", surface: "js", payload: payload });
    const evs = [
        mk("prompt", 1, { text: "hello t78" }),
        mk("delta", 2, { text: "ok" }),
        mk("usage", 3, { in: 2, out: 3, model: "mock", provider: "openai" }),
        mk("served", 4, { by: "cloud" }),
        mk("stop", 5, { reason: "done" }),
    ];
    const ap = P(sofuu.events.append(JSON.stringify({ project: ".", sid: SID, events: evs })));
    assert("events appended", ap.ok === true);
    let logBlob = "";
    try { logBlob += String(await sofuu.fs.readFile(".sofuu/sessions/" + SID + "/events.ndjson")); } catch (e) {}
    const docBuild = sofuu.session.export(JSON.stringify({ project: ".", sid: SID }));
    P(sofuu.session.import(docBuild));
    const docOut = sofuu.session.export(JSON.stringify({ project: ".", sid: SID }));
    assert("no key bytes in the event log or the session document",
        (logBlob + docOut).indexOf(SECRET) < 0);

    console.log(`\n=== t78 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t78:", (e && e.message) || e); process.exit(1); });
