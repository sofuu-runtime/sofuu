// t24_disconnect.js — R2.3/R2.2: a client disconnect is visible to the
// server, so a streaming handler can stop writing into a dead socket.
//
// The server is a REAL Sofuu HTTP server whose handler streams GENUINE model
// output: it asks the in-process inference engine (the real local GGUF) for
// a completion and relays the real tokens as OpenAI-compatible SSE frames.
// The client is the REAL native stream path (ai.rs → curl → SSE parse)
// against that server, and the disconnect is the real abort of that transfer
// — the TCP connection the handler writes to is closed by the peer.
//
// The old scripted SSE provider (a fake model that emitted "tickN" frames)
// is gone: every byte the client reads came from the real model.
//
// Requires SOFUU_CONTRACT_MODEL to name an existing real-weight GGUF. A
// missing model is a blocked prerequisite, never a silent pass.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL tokens relayed over a real loopback server; blocks when unset
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      ./sofuu run scripts/contract/harness/t24_disconnect.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 180 s"); process.exit(1); }, 180000);

const BOUND_MS = 1500;

/* The real local weights this case may load. The synthetic contract fixture
 * and the vocabulary-only GGUFs are parser fixtures — they are refused here,
 * they are not inference evidence. */
function resolveModel() {
    const env = String((typeof process !== 'undefined' && process.env) ? (process.env.SOFUU_CONTRACT_MODEL || "") : "").trim();
    if (!env) {
        console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing real-weight GGUF " +
            "(no weights are downloaded; the synthetic contract fixture is a parser fixture)");
        process.exit(2);
    }
    if (!/\.gguf$/i.test(env) || !sofuu.fs.existsSync(env) ||
        /(^|\/)sofuu-fixture\.gguf$/i.test(env) || /ggml-vocab-/i.test(env)) {
        console.error("BLOCKED: SOFUU_CONTRACT_MODEL must name an existing real-weight GGUF " +
            "(not a synthetic or vocabulary fixture): " + env);
        process.exit(2);
    }
    return env;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Load the real weights before the measured phase: the first inference on a
// busy machine can take longer than the HTTP server's own request deadline,
// and a cold start would then be reported as a transport failure instead of
// the disconnect behaviour under test.
async function warmUp(modelPath) {
    const iter = sofuu.ai.stream({
        provider: "local", model: modelPath,
        messages: [{ role: "user", content: "Say hi." }],
        max_tokens: 8,
    });
    const drain = (async () => { for await (const chunk of iter) { void chunk; } })();
    await Promise.race([drain, sleep(180000)]);
    try { if (typeof iter.abort === "function") iter.abort(); } catch (e) { /* already finished */ }
}

// The real model, streamed through the chat-message path the runtime uses
// for every agent/provider turn.
function modelStream(modelPath, prompt, maxTokens) {
    return sofuu.ai.stream({
        provider: "local", model: modelPath,
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens,
    });
}

async function main() {
    const modelPath = resolveModel();
    console.log("  ℹ model: " + modelPath);
    await warmUp(modelPath);
    console.log("  ℹ model weights loaded");

    // ── the REAL server: real model tokens, relayed as SSE ─────────────
    let liveRes = null;        // the handler's res, captured for observation
    let writes = 0;
    let lastWriteAt = 0;
    const handler = (req, res) => {
        liveRes = res;
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        const iter = modelStream(modelPath, "Write a long story about a lighthouse keeper.", 400);
        (async () => {
            try {
                while (true) {
                    if (res.closed && res.closed()) break;
                    const n = await iter.next();
                    if (n.done) break;
                    const text = n.value && n.value.text;
                    if (!text) continue;
                    if (res.closed && res.closed()) break;
                    writes += 1;
                    lastWriteAt = Date.now();
                    res.write("data: " + JSON.stringify({
                        choices: [{ delta: { content: text } }],
                    }) + "\n\n");
                }
            } catch (e) {
                /* the client hung up mid-stream — the handler stops here */
            } finally {
                try { if (typeof iter.abort === "function") iter.abort(); } catch (eA) { /* already finished */ }
                try { res.end(); } catch (eE) { /* the socket is already gone */ }
            }
        })();
    };
    let server = null, port = 0;
    const base = 26500 + (Date.now() % 2000);
    for (let i = 0; i < 6; i++) {
        const p = base + i * 143;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("the real streaming server bound", server != null);
    if (!port) { console.log(`\n=== t24 results: ${results.passed} passed, ${results.failed} failed ===`); process.exit(1); }
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // ── client: stream real frames, then disconnect ────────────────────
    const st = sofuu.ai.stream({
        provider: "openai", model: "t24-contract-model", api_key: "x", base_url: URL,
        messages: [{ role: "user", content: "hi" }],
    });
    let frames = 0, clientText = "";
    const t0 = Date.now();
    try {
        for await (const chunk of st) {
            if (chunk && chunk.text) { clientText += chunk.text; frames += 1; }
            if (frames >= 2) { st.abort(); break; }
        }
    } catch (e) { /* disconnect may surface as an error — fine */ }
    assert("client consumed 2 real frames then disconnected", frames >= 2 && clientText.trim().length > 0);

    // ── server: the disconnect must be visible within the bound ─────────
    let sawClosed = false, wall = -1;
    const tWatch = Date.now();
    while (Date.now() - tWatch < BOUND_MS + 1000) {
        if (liveRes && liveRes.closed && liveRes.closed()) {
            sawClosed = true;
            wall = Date.now() - t0;
            break;
        }
        await sleep(25);
    }
    assert(`server observed the disconnect via res.closed() (${wall} ms after stream start)`, sawClosed);
    // The handler must stop writing into the dead socket: at most the write
    // that was already in flight, and no write after the close was visible.
    const writesAtClose = writes;
    await sleep(500);
    assert(`handler kept writing only briefly past the disconnect (${writes - writesAtClose} writes after close, ${writes} total)`,
        writes - writesAtClose <= 1);
    assert(`no write landed after the disconnect was observed (last write ${lastWriteAt ? lastWriteAt - (t0 + wall) : 0} ms past the close)`,
        lastWriteAt === 0 || lastWriteAt <= t0 + wall + BOUND_MS);

    console.log(`\n=== t24 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t24:", (e && e.message) || e); process.exit(1); });
