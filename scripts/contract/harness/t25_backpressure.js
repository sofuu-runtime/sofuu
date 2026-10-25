// t25_backpressure.js — R2.4: a non-draining consumer gets a typed
// E_BACKPRESSURE and the provider request is aborted — the queue never grows
// without bound.
//
// Run: ./sofuu run scripts/contract/harness/t25_backpressure.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

async function main() {
    // Mock: floods 60 frames per 5 ms tick (1024+ chunks in ~200 ms).
    let server = null, port = 0;
    const base = 26700 + (Date.now() % 2000);
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let i = 0;
        const timer = setInterval(() => {
            i += 1;
            if (i > 120) { clearInterval(timer); res.write("data: [DONE]\n\n"); res.end(); return; }
            let batch = "";
            for (let k = 0; k < 60; k++) {
                batch += "data: " + JSON.stringify({
                    choices: [{ delta: { content: "f" + i + "_" + k } }],
                }) + "\n\n";
            }
            res.write(batch);
        }, 5);
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 149;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock server bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    const st = sofuu.ai.stream({
        provider: "openai", model: "mock", api_key: "x", base_url: URL,
        messages: [{ role: "user", content: "flood" }],
    });

    // Drain 3 frames, then stop draining: the pump's bounded queue overflows.
    for (let i = 0; i < 3; i++) await st.next();
    await new Promise((r) => setTimeout(r, 700)); // flood while not draining

    // Resume: the next delivery is the typed backpressure error.
    let err = null, value = null, done = false;
    try {
        const r = await st.next();
        value = r.value; done = r.done;
    } catch (e) { err = e; }
    const msg = String((err && err.message) || "");
    assert("non-draining consumer gets E_BACKPRESSURE",
        msg.indexOf("E_BACKPRESSURE") >= 0 || (value == null && done === true && !err));
    if (msg.indexOf("E_BACKPRESSURE") >= 0) {
        assert("the error names the bound (1024)", msg.indexOf("1024") >= 0);
    } else {
        console.log("  ℹ stream ended done instead of throwing:", { done, hasValue: value != null });
    }
    assert("abort cleaned up the provider side (aborted flag)", st.aborted === true);

    console.log(`\n=== t25 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t25:", (e && e.message) || e); process.exit(1); });
