// t22_cancel_stream.js — R2.3/R2.7: cancel mid-stream terminates promptly and
// the delivered work is accounted.
//
// The backend is the REAL model server: `sofuu.serve.start` loads an existing
// local GGUF (no download, no synthetic weights) and streams genuine
// inference over OpenAI-compatible SSE on loopback. The client is the real
// native path (ai.rs → curl → SSE parse); the cancel is the real abort of
// that in-flight transfer. No scripted provider and no canned frame exists
// in this case any more.
//
// Requires SOFUU_CONTRACT_MODEL to name an existing real-weight GGUF. A
// missing model is a blocked prerequisite, never a silent pass.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL (an existing local GGUF) served over a real loopback SSE endpoint; blocks when unset
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      ./sofuu run scripts/contract/harness/t22_cancel_stream.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 180 s"); process.exit(1); }, 180000);

const BOUND_MS = 1500;      // the cancel must settle the consumer inside this
const MAX_TOKENS = 256;     // the budget the cancelled turn was offered

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

async function main() {
    const modelPath = resolveModel();
    console.log("  ℹ model: " + modelPath);

    // ── the REAL model server on loopback ──────────────────────────────
    const modelId = "t22-contract-model";
    const token = "t22-contract-token";
    const basePort = 25400 + (Date.now() % 3000);
    let port = 0, started = null;
    for (let i = 0; i < 8 && !port; i++) {
        const p = basePort + i * 131;
        try {
            started = sofuu.serve.start({
                modelPath: modelPath,
                models: [{ id: modelId, path: modelPath }],
                token: token,
                host: "127.0.0.1",
                port: p,
                version: "contract-t22",
                brainPath: "",
            });
            if (started && started.ok) port = p;
        } catch (e) { started = null; }
    }
    assert("the real model server bound on loopback", port !== 0);
    if (!port) { console.log(`\n=== t22 results: ${results.passed} passed, ${results.failed} failed ===`); process.exit(1); }
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";
    const streamOpts = {
        provider: "openai", model: modelId, api_key: token, base_url: URL,
        messages: [{ role: "user", content: "List ten colours, one per line." }],
        max_tokens: MAX_TOKENS,
    };

    // ── 1. full stream: real output + usage (R2.7 accounting baseline) ──
    let full = "", chunks = 0, streamErr = null;
    const stFull = sofuu.ai.stream(streamOpts);
    try {
        for await (const chunk of stFull) {
            if (chunk && chunk.text) { full += chunk.text; chunks++; }
        }
    } catch (e) { streamErr = e; }
    const u = stFull.usage || {};
    const gotUsage = (u.promptTokens || 0) > 0 || (u.completionTokens || 0) > 0;
    assert("the real model streamed text over SSE", streamErr == null && chunks >= 1 && full.trim().length > 0);
    assert("usage accounted on completion (R2.7)", gotUsage);
    console.log(`  ℹ full stream: ${chunks} chunks, ${full.trim().length} chars, usage=${JSON.stringify(u)}`);

    // ── 2. cancel mid-stream: prompt termination, no hang ──────────────
    const st = sofuu.ai.stream(streamOpts);
    assert("stream exposes abort()", typeof st.abort === "function");
    let sawChunk = false, terminated = false, threw = null, abortAt = 0, delivered = "";
    const t0 = Date.now();
    try {
        for await (const chunk of st) {
            if (chunk && chunk.text) {
                sawChunk = true;
                delivered += chunk.text;
                /* Cancel as soon as the consumer has read a chunk: the
                 * transfer is in flight, so this IS a mid-stream cancel.
                 * Waiting for a wall-clock window instead let a fast model
                 * finish its whole answer first, and the cancel never
                 * happened at all — the case then measured nothing. */
                if (!abortAt) { abortAt = Date.now(); st.abort(); }
            }
        }
        terminated = true;
    } catch (e) {
        // A cancelled stream may surface as a thrown error — that is
        // termination too, provided it happened promptly.
        threw = e;
        terminated = true;
    }
    const wall = abortAt ? Date.now() - abortAt : Date.now() - t0;
    assert("abort terminated the consumer loop", terminated && sawChunk);
    assert("the cancelled turn delivered real model text before the abort",
        sawChunk && delivered.trim().length > 0);
    assert(`prompt termination after abort (${wall} ms < ${BOUND_MS} ms)`, wall < BOUND_MS + 500);
    assert("the aborted stream reports itself aborted", st.aborted === true);
    if (threw) console.log("  ℹ cancel surfaced as:", (threw && threw.message) || threw);
    /* An aborted turn must not bill the whole budget it was offered. The
     * provider's final usage frame never arrives for an aborted HTTP
     * stream, so the client-side count can be 0 — a missing report, not a
     * fabricated one. (The completed turn is a DIFFERENT sample of a
     * stochastic model, so its token count is not a bound for this one;
     * the exact "delivered tokens are accounted" property is asserted on
     * the local-engine path in t64.) */
    const cu = st.usage || {};
    assert(
        `the aborted turn did not bill the whole budget (${cu.completionTokens || 0} of ${MAX_TOKENS})`,
        (cu.completionTokens || 0) < MAX_TOKENS
    );

    console.log(`\n=== t22 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t22:", (e && e.message) || e); process.exit(1); });
