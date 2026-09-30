// tests/ai_stream_test.js — ai.stream over a LOCAL mock provider
// (assert-based, A0.2).
// P2-26 (AUDIT-2026-09-01): the old version replaced sofuu.ai.stream
// with its own JS async-iterator and asserted on its own chunks — a
// tautology that passed with a fully broken runtime. This one runs the
// REAL native streaming path (ai.rs → curl → SSE parse) against an
// in-process mock endpoint:
//   1. streamed text arrives as ordered chunks + usage at end
//   2. think-tag content routes to chunk.think, not chunk.text
//   3. the stream object exposes abort()
// Run:  ./sofuu run tests/ai_stream_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== ai.stream() Test ===\n");

assert("sofuu.ai.stream is function", typeof sofuu.ai.stream === "function");

function sse(res, frames) {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    for (const f of frames) res.write("data: " + f + "\n\n");
    res.end();
}

const handler = (req, res) => {
    let parsed = {};
    try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
    const msgs = parsed.messages || [];
    let last = "";
    for (let i = msgs.length - 1; i >= 0; i--) {
        if (String(msgs[i].role) === "user") { last = String(msgs[i].content || ""); break; }
    }
    if (last.indexOf("THINKY") >= 0) {
        sse(res, [
            JSON.stringify({ choices: [{ delta: { reasoning_content: "internal pondering about the query" } }] }),
            JSON.stringify({ choices: [{ delta: { content: "visible answer" } }] }),
            JSON.stringify({ usage: { prompt_tokens: 7, completion_tokens: 5 } }),
            "[DONE]",
        ]);
        return;
    }
    sse(res, [
        JSON.stringify({ choices: [{ delta: { content: "Hello" } }] }),
        JSON.stringify({ choices: [{ delta: { content: " " } }] }),
        JSON.stringify({ choices: [{ delta: { content: "world" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "!" } }] }),
        JSON.stringify({ usage: { prompt_tokens: 2, completion_tokens: 4 } }),
        "[DONE]",
    ]);
};

async function main() {
    let server = null, port = 0;
    const base = 25200 + (Date.now() % 4000);
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

    // ── 1. streamed text + usage ──────────────────────────────────
    const st = sofuu.ai.stream(mk("plain hello"));
    assert("stream exposes abort()", typeof st.abort === "function");
    let full = "";
    let sawThink = false;
    for await (const chunk of st) {
        if (chunk && chunk.text) full += chunk.text;
        if (chunk && chunk.think) sawThink = true;
    }
    assert("chunks concatenate to 'Hello world!'", full === "Hello world!");
    assert("no think content on the plain stream", !sawThink);
    const u = st.usage || {};
    assert("usage reports promptTokens 2", (u.promptTokens || 0) === 2);
    assert("usage reports completionTokens 4", (u.completionTokens || 0) === 4);

    // ── 2. think content routes to .think ─────────────────────────
    const st2 = sofuu.ai.stream(mk("THINKY hello"));
    let t2 = "", k2 = "";
    for await (const chunk of st2) {
        if (chunk && chunk.text) t2 += chunk.text;
        if (chunk && chunk.think) k2 += chunk.think;
    }
    assert("think text separated from the answer", k2.indexOf("internal pondering") >= 0);
    assert("answer excludes the think block", t2.indexOf("internal pondering") < 0);
    assert("answer keeps visible text", t2.indexOf("visible answer") >= 0);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All ai.stream tests PASSED (local mock provider)");
}

main().catch(err => {
    console.error("FAIL ai.stream test:", (err && err.message) || err);
    process.exit(1);
});
