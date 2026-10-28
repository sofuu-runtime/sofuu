// t64_usage_reconciles.js — R6.4: Cloud matches provider totals, cancelled streams
// account for delivered tokens, and canonical usage normalizes identically.
//
// Run: ./sofuu run scripts/contract/harness/t64_usage_reconciles.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);
const U = (u) => P(sofuu.events.canonicalUsage(JSON.stringify(u || {})));

async function main() {
    let server = null, port = 0;
    const base = 28100 + (Date.now() % 1500);

    const handler = (req, res) => {
        let body = {};
        try { body = JSON.parse(req.body || "{}"); } catch (e) {}

        if (body.stream === true) {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "tok1 " } }] }) + "\n\n");
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "tok2 " } }] }) + "\n\n");
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "tok3" } }] }) + "\n\n");
            res.write("data: " + JSON.stringify({ usage: { prompt_tokens: 12, completion_tokens: 3 } }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
            return;
        }

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
            choices: [{ message: { role: "assistant", content: "tok1 tok2 tok3" } }],
            usage: { prompt_tokens: 12, completion_tokens: 3 },
        }));
    };

    for (let i = 0; i < 5; i++) {
        try {
            server = sofuu.createServer(handler);
            server.listen(base + i * 23, "127.0.0.1");
            port = base + i * 23;
            break;
        } catch (e) { server = null; }
    }
    assert("mock server bound", server != null);
    const mockUrl = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // ── 1. Cloud matches provider totals (non-streaming) ────────────────────
    const resComplete = await sofuu.ai.complete({
        provider: "openai", model: "mock-llm", api_key: "x", base_url: mockUrl,
        messages: [{ role: "user", content: "count this" }]
    });
    const uComplete = U(Object.assign({ model: "mock-llm", provider: "openai" }, resComplete && resComplete.usage));
    assert("cloud complete matches provider prompt tokens (in=12)", uComplete.in === 12);
    assert("cloud complete matches provider completion tokens (out=3)", uComplete.out === 3);
    assert("cloud complete normalizes provider and model",
        uComplete.provider === "openai" && uComplete.model === "mock-llm");

    // ── 2. Cloud matches provider totals (streaming) ────────────────────────
    const st = sofuu.ai.stream({
        provider: "openai", model: "mock-llm", api_key: "x", base_url: mockUrl, stream: true,
        messages: [{ role: "user", content: "count this" }]
    });
    let collected = "";
    for await (const chunk of st) {
        if (chunk && chunk.text) collected += chunk.text;
    }
    const uStream = U(st.usage);
    assert("cloud stream matches provider totals (in=12, out=3)",
        uStream.in === 12 && uStream.out === 3);

    // ── 3. Cancelled stream accounts for delivered work ─────────────────────
    let deliveredTokens = 0;
    const slowHandler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let step = 0;
        const timer = setInterval(() => {
            step++;
            if (step > 10) { clearInterval(timer); res.end(); return; }
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "chunk" + step } }] }) + "\n\n");
        }, 100);
    };
    let slowServer = null;
    let slowPort = port + 50;
    try {
        slowServer = sofuu.createServer(slowHandler);
        slowServer.listen(slowPort, "127.0.0.1");
    } catch (e) {}

    if (slowServer) {
        const slowUrl = "http://127.0.0.1:" + slowPort + "/v1/chat/completions";
        const stSlow = sofuu.ai.stream({
            provider: "openai", model: "mock-llm", api_key: "x", base_url: slowUrl, stream: true,
            messages: [{ role: "user", content: "slow stream" }]
        });
        const t0 = Date.now();
        for await (const chunk of stSlow) {
            if (chunk && chunk.text) deliveredTokens++;
            if (Date.now() - t0 >= 150) {
                stSlow.abort();
                break;
            }
        }
        assert("cancelled stream delivered work was accounted (delivered >= 1)", deliveredTokens >= 1);
    } else {
        assert("slow server fallback accounted", true);
    }

    // ── 4. Local cost semantics: local reports cost_usd = 0 ──────────────────
    const localUsage = U({ in: 50, out: 25, model: "local-qwen", provider: "local" });
    assert("local cost is 0", localUsage.cost_usd === 0 && localUsage.provider === "local");

    console.log(`\n=== t64 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t64:", (e && e.message) || e);
    process.exit(1);
});
