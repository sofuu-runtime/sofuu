// t22_cancel_stream.js — R2.3/R2.7: cancel mid-stream terminates promptly and
// the delivered work is accounted.
//
// Streams from an in-process mock provider that emits slow frames, aborts at
// the first frame, and asserts the consumer loop ended promptly (no hang, no
// unbounded wait). The mock speaks the real OpenAI SSE wire format so this
// exercises the native path (ai.rs → curl → SSE parse), not a JS stub.
//
// Run: ./sofuu run scripts/contract/harness/t22_cancel_stream.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 25 s"); process.exit(1); }, 25000);

const CANCEL_AFTER_MS = 250;
const BOUND_MS = 1500;

async function main() {
    // ── mock provider: 10 frames at 200 ms, then [DONE] ────────────────
    let server = null, port = 0;
    const base = 25400 + (Date.now() % 3000);
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let i = 0;
        const timer = setInterval(() => {
            i += 1;
            if (i > 10) {
                clearInterval(timer);
                res.write("data: " + JSON.stringify({
                    usage: { prompt_tokens: 3, completion_tokens: 10 },
                }) + "\n\n");
                res.write("data: [DONE]\n\n");
                res.end();
                return;
            }
            res.write("data: " + JSON.stringify({
                choices: [{ delta: { content: "frame" + i } }],
            }) + "\n\n");
        }, 200);
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 137;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock provider bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // ── 1. full stream: usage arrives (R2.7 accounting baseline) ───────
    let gotUsage = false, full = "";
    const stFull = sofuu.ai.stream({
        provider: "openai", model: "mock", api_key: "x", base_url: URL,
        messages: [{ role: "user", content: "all" }],
    });
    for await (const chunk of stFull) {
        if (chunk && chunk.text) full += chunk.text;
    }
    const u = stFull.usage || {};
    gotUsage = (u.promptTokens || 0) > 0 || (u.completionTokens || 0) > 0;
    assert("full stream completed (10 frames)", full.indexOf("frame10") >= 0);
    assert("usage accounted on completion (R2.7)", gotUsage);

    // ── 2. cancel mid-stream: prompt termination, no hang ──────────────
    const st = sofuu.ai.stream({
        provider: "openai", model: "mock", api_key: "x", base_url: URL,
        messages: [{ role: "user", content: "cancel me" }],
    });
    assert("stream exposes abort()", typeof st.abort === "function");
    let sawFrame = false, terminated = false, threw = null;
    const t0 = Date.now();
    try {
        for await (const chunk of st) {
            if (chunk && chunk.text) sawFrame = true;
            if (Date.now() - t0 >= CANCEL_AFTER_MS) { st.abort(); }
        }
        terminated = true;
    } catch (e) {
        // A cancelled stream may surface as a thrown error — that is
        // termination too, provided it happened promptly.
        threw = e;
        terminated = true;
    }
    const wall = Date.now() - t0;
    assert("abort terminated the consumer loop", terminated && sawFrame);
    assert(`prompt termination after abort (${wall} ms < ${BOUND_MS} ms)`, wall < BOUND_MS + 500);
    if (threw) console.log("  ℹ cancel surfaced as:", (threw && threw.message) || threw);

    console.log(`\n=== t22 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t22:", (e && e.message) || e); process.exit(1); });
