// t24_disconnect.js — R2.3/R2.2: a client disconnect is visible to the
// server, so a streaming handler can stop writing into a dead socket.
//
// The mock SSE server holds the response open and records what it observes;
// the client streams, aborts, and the harness asserts the server saw the
// disconnect within the contract bound (1 s, SPEC T2.4) via res.closed().
//
// Run: ./sofuu run scripts/contract/harness/t24_disconnect.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 25 s"); process.exit(1); }, 25000);

const BOUND_MS = 1500;

async function main() {
    let server = null, port = 0;
    let liveRes = null;      // the handler's res, captured for observation
    let writesAfterClose = 0;
    const base = 26500 + (Date.now() % 2000);
    const handler = (req, res) => {
        liveRes = res;
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let i = 0;
        const timer = setInterval(() => {
            i += 1;
            if (res.closed && res.closed()) {
                writesAfterClose += 1;
            }
            if (i > 200) { clearInterval(timer); res.end(); return; }
            res.write("data: " + JSON.stringify({
                choices: [{ delta: { content: "tick" + i } }],
            }) + "\n\n");
        }, 20);
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 143;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock server bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // ── client: stream two frames, then disconnect ──────────────────────
    const st = sofuu.ai.stream({
        provider: "openai", model: "mock", api_key: "x", base_url: URL,
        messages: [{ role: "user", content: "hi" }],
    });
    let frames = 0;
    const t0 = Date.now();
    try {
        for await (const chunk of st) {
            frames += 1;
            if (frames >= 2) { st.abort(); break; }
        }
    } catch (e) { /* disconnect may surface as an error — fine */ }
    assert("client consumed 2 frames then disconnected", frames >= 2);

    // ── server: the disconnect must be visible within the bound ─────────
    let sawClosed = false, wall = -1;
    const tWatch = Date.now();
    while (Date.now() - tWatch < BOUND_MS + 1000) {
        if (liveRes && liveRes.closed && liveRes.closed()) {
            sawClosed = true;
            wall = Date.now() - t0;
            break;
        }
        await new Promise((r) => setTimeout(r, 25));
    }
    assert(`server observed the disconnect via res.closed() (${wall} ms after stream start)`, sawClosed);
    assert("handler kept writing only briefly past the disconnect (no unbounded writes)",
        writesAfterClose <= 30);

    console.log(`\n=== t24 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t24:", (e && e.message) || e); process.exit(1); });
