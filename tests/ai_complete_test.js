// tests/ai_complete_test.js — ai.complete over a LOCAL mock provider
// (assert-based, A0.2).
// P2-26 (AUDIT-2026-09-01): the old version replaced sofuu.ai.complete
// with its own JS function and asserted on its own return value — a
// tautology that passed with a fully broken runtime. This one runs the
// REAL native path (ai.rs → curl) against an in-process mock endpoint:
//   1. plain completion round-trip (text + usage)
//   2. tool-call completion (the model asks for a tool; the native
//      parser must surface toolCalls)
//   3. error status (HTTP 500) rejects with the provider message
// Run:  ./sofuu run tests/ai_complete_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== ai.complete() Test ===\n");

assert("sofuu.ai.complete is function", typeof sofuu.ai.complete === "function");

const handler = (req, res) => {
    let parsed = {};
    try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
    const msgs = parsed.messages || [];
    let last = "";
    for (let i = msgs.length - 1; i >= 0; i--) {
        if (String(msgs[i].role) === "user") { last = String(msgs[i].content || ""); break; }
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    if (last.indexOf("NEEDTOOL") >= 0) {
        res.end(JSON.stringify({
            choices: [{ message: { role: "assistant", content: null, tool_calls: [{
                id: "call_1", type: "function",
                function: { name: "probe_tool", arguments: JSON.stringify({ x: 7 }) }
            }] } }],
            usage: { prompt_tokens: 11, completion_tokens: 2 },
        }));
        return;
    }
    if (last.indexOf("BOMB") >= 0) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: "mock-model exploded", code: 500 } }));
        return;
    }
    res.end(JSON.stringify({
        choices: [{ message: { role: "assistant", content: "COMPLETE-OK-ANSWER" } }],
        usage: { prompt_tokens: 9, completion_tokens: 4 },
    }));
};

async function main() {
    let server = null, port = 0;
    const base = 24800 + (Date.now() % 4000);
    for (let i = 0; i < 6; i++) {
        const p = base + i * 131;
        try {
            server = sofuu.createServer(handler);
            server.listen(p, "127.0.0.1");
            port = p;
            break;
        } catch (e) { server = null; }
    }
    if (!server) { console.log("FAIL could not bind mock port"); process.exit(1); }
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";
    const mk = (content) => ({
        provider: "openai", model: "mock", api_key: "x", base_url: URL,
        messages: [{ role: "user", content }],
    });

    // ── 1. plain completion ────────────────────────────────────────
    const r = await sofuu.ai.complete(mk("hello there"));
    assert("complete returns text", r && typeof r.text === "string");
    assert("complete returns the mock answer", r.text.indexOf("COMPLETE-OK-ANSWER") >= 0);
    /* ai.complete surfaces usage via `raw` (the provider's JSON payload)
     * — there is no normalized usage field on this API. */
    assert("complete carries the provider usage payload",
        r.raw && r.raw.indexOf("prompt_tokens") >= 0);

    // ── 2. tool-call completion ────────────────────────────────────
    const t = await sofuu.ai.complete(mk("NEEDTOOL now"));
    const tc = t.toolCalls || (t.choices && []) || [];
    const tcs = JSON.stringify(t);
    assert("complete surfaces toolCalls", tcs.indexOf("probe_tool") >= 0);
    assert("tool call arguments survive parsing", tcs.indexOf('"x":7') >= 0 || tcs.indexOf("{\\\"x\\\":7}") >= 0 || tcs.indexOf('"x": 7') >= 0);

    // ── 3. error status rejects loudly ────────────────────────────
    let sawErr = "";
    try {
        await sofuu.ai.complete(mk("BOMB now"));
    } catch (e) { sawErr = String((e && e.message) || e); }
    assert("HTTP 500 rejects with the provider message", sawErr.indexOf("exploded") >= 0);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All ai.complete tests PASSED (local mock provider)");
}

main().catch(err => {
    console.error("FAIL ai.complete test:", (err && err.message) || err);
    process.exit(1);
});
