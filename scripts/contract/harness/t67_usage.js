// t67_usage.js — R6.4/T6.7: usage reconciles — every surface reports the same
// shape, provider totals match, local cost is 0.
//
// Run: ./sofuu run scripts/contract/harness/t67_usage.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 25 s"); process.exit(1); }, 25000);

const P = (s) => JSON.parse(s);
const U = (u) => P(sofuu.events.canonicalUsage(JSON.stringify(u || {})));

async function main() {
    // Mock: SSE for streamed requests, a JSON completion body (with usage)
    // otherwise — so both surfaces are exercised against the same provider.
    let server = null, port = 0;
    const base = 26900 + (Date.now() % 2000);
    const handler = (req, res) => {
        let body = {};
        try { body = JSON.parse(req.body || "{}"); } catch (e) {}
        if (body.stream === true) {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "hi" } }] }) + "\n\n");
            res.write("data: " + JSON.stringify({ usage: { prompt_tokens: 5, completion_tokens: 7 } }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
            return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
            choices: [{ message: { role: "assistant", content: "hi" } }],
            usage: { prompt_tokens: 5, completion_tokens: 7 },
        }));
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 151;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock provider bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";
    const mk = (extra) => Object.assign(
        { provider: "openai", model: "mock", api_key: "x", base_url: URL }, extra);

    // ── 1. ai.complete reports usage (the last surface without a shape) ──
    const done = await sofuu.ai.complete(mk({ messages: [{ role: "user", content: "hi" }] }));
    const u1 = U(done && done.usage);
    assert("ai.complete reports usage", (done && done.usage) != null);
    assert("provider totals match (in=5)", u1.in === 5);
    assert("provider totals match (out=7)", u1.out === 7);
    assert("cloud cost carried (mock sets none → 0)", u1.cost_usd === 0);

    // ── 2. ai.stream usage matches the same provider totals ─────────────
    const st = sofuu.ai.stream(mk({ stream: true, messages: [{ role: "user", content: "hi" }] }));
    let text = "";
    for await (const chunk of st) { if (chunk && chunk.text) text += chunk.text; }
    const u2 = U(st.usage);
    assert("stream usage matches provider totals (in=5, out=7)",
        u2.in === 5 && u2.out === 7);

    // ── 3. agent usage normalizes through the same shape ────────────────
    const agentU = U({ promptTokens: 5, completionTokens: 7, model: "mock", provider: "openai" });
    assert("agent shape normalizes identically",
        agentU.in === u2.in && agentU.out === u2.out && agentU.model === "mock");

    // ── 4. one canonical shape: in/out/model/provider/cost_usd ──────────
    const keys = Object.keys(u1).sort().join(",");
    assert("canonical keys", keys === "cost_usd,in,model,out,provider");

    console.log(`\n=== t67 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t67:", (e && e.message) || e); process.exit(1); });
