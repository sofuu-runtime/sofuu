// tests/ai_stream_error_frame_test.js — ai.stream failure surfacing
// (assert-based, A0.2). Regression test for the chat "no response" bug:
// gateways (OpenRouter et al.) can return HTTP 200 and then report the
// upstream failure as a `data: {"error": {...}}` SSE frame. The stream
// must reject with the provider's message — not complete silently empty.
//
// Covers, over a local mock (real curl path, no network, no keys):
//   1. in-stream error frame (200 + error object + [DONE]) → reject,
//      message carries the frame's code + text
//   2. HTTP 429 status + JSON body → reject with "HTTP 429" + detail
//   3. empty stream (200 + [DONE] only) → completes with 0 chunks,
//      does NOT throw (the agent loop owns the empty-stream policy)
//   4. normal stream still delivers text (parser branch sanity)
//
// Run:  ./sofuu run tests/ai_stream_error_frame_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== ai.stream error surfacing test ===\n");

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
    if (last.indexOf("ERRFRAME") >= 0) {
        sse(res, [
            JSON.stringify({ error: { message: "mock-model upstream failed mid-stream", code: 429 } }),
            "[DONE]",
        ]);
        return;
    }
    if (last.indexOf("STATUS429") >= 0) {
        res.writeHead(429, { "Content-Type": "application/json" });
        res.send(JSON.stringify({ error: { message: "temporarily rate-limited upstream", code: 429 } }));
        return;
    }
    if (last.indexOf("EMPTY") >= 0) {
        sse(res, ["[DONE]"]);
        return;
    }
    sse(res, [
        JSON.stringify({ choices: [{ delta: { content: "OK-TEXT" } }] }),
        JSON.stringify({ usage: { prompt_tokens: 3, completion_tokens: 2 } }),
        "[DONE]",
    ]);
};

async function collect(opts) {
    const st = sofuu.ai.stream(opts);
    const parts = [];
    let err = null;
    try {
        for await (const ch of st) { if (ch && ch.text) parts.push(ch.text); }
    } catch (e) { err = e; }
    return { text: parts.join(""), err: String((err && err.message) || err || "") };
}

async function main() {
    let server = null, port = 0;
    const base = 24000 + (Date.now() % 4000);
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

    /* 1. in-stream error frame on a 200 response */
    const ef = await collect(mk("ERRFRAME hello"));
    assert("error frame: stream rejects (no silent empty)", ef.err !== "");
    assert("error frame: message carries frame code 429", ef.err.indexOf("429") >= 0);
    assert("error frame: message carries provider text", ef.err.indexOf("upstream failed mid-stream") >= 0);
    assert("error frame: no partial answer text", ef.text === "");

    /* 2. HTTP 429 status */
    const s429 = await collect(mk("STATUS429 hello"));
    assert("status 429: stream rejects", s429.err !== "");
    assert("status 429: message says HTTP 429", s429.err.indexOf("HTTP 429") >= 0);
    assert("status 429: message carries body detail", s429.err.indexOf("rate-limited upstream") >= 0);

    /* 3. empty stream completes without throwing */
    const empty = await collect(mk("EMPTY hello"));
    assert("empty stream: no error thrown", empty.err === "");
    assert("empty stream: zero text", empty.text === "");

    /* 4. normal stream unaffected */
    const ok = await collect(mk("PLAIN hello"));
    assert("normal stream: no error", ok.err === "");
    assert("normal stream: text delivered", ok.text === "OK-TEXT");

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All ai.stream error-surfacing tests PASSED");
}

main().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
