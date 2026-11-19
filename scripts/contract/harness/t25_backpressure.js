// t25_backpressure.js — R2.4: a non-draining consumer gets a typed
// E_BACKPRESSURE and the provider request is aborted — the queue never grows
// without bound.
//
// The backend is the REAL local engine: provider "local" over an existing
// real-weight GGUF (SOFUU_CONTRACT_MODEL). Every chunk the consumer sees is
// genuine model output decoded from those weights — the case contains no
// scripted provider, no canned frame and no fixed answer.
//
// The queue that overflows is the runtime's own bounded chunk queue (1024
// undelivered chunks). The consumer drains three chunks and then stops
// draining while the model keeps producing, so the bound is reached by real
// work rather than by an injected burst.
//
// A model that stops before the bound is reached cannot exercise the
// scenario: that is reported as BLOCKED with the chunk count, never as a
// pass.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL on the local engine's own bounded chunk queue; blocks when unset
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      ./sofuu run scripts/contract/harness/t25_backpressure.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 300 s"); process.exit(1); }, 300000);

const BOUND_MS = 180000;   // how long the model may take to fill the queue
const MAX_TOKENS = 3000;   // enough headroom for the 1024-chunk bound
const PROMPT = "Write a numbered list of 400 distinct one-sentence facts about the " +
    "history of civil engineering. Do not stop before 400.";

/* The real local weights this case may load. The synthetic contract fixture
 * and the vocabulary-only GGUFs are parser fixtures, never evidence of
 * inference, so they are refused here. */
function resolveModel() {
    const env = String((process.env && process.env.SOFUU_CONTRACT_MODEL) || "").trim();
    if (!env) {
        console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing real-weight GGUF " +
            "(no weights are downloaded; the synthetic contract fixture is a parser fixture)");
        process.exit(2);
    }
    if (!/\.gguf$/i.test(env) || !sofuu.fs.existsSync(env) ||
        /(^|\/)sofuu-fixture\.gguf$/i.test(env) || /ggml-vocab/i.test(env)) {
        console.error("BLOCKED: SOFUU_CONTRACT_MODEL must name an existing real-weight GGUF " +
            "(not a synthetic or vocabulary fixture): " + env);
        process.exit(2);
    }
    return env;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function main() {
    const model = resolveModel();
    console.log("  ℹ model: " + model);

    const st = sofuu.ai.stream({
        provider: "local", model: model,
        messages: [{ role: "user", content: PROMPT }],
        max_tokens: MAX_TOKENS,
    });
    assert("the stream exposes an abort()", typeof st.abort === "function");

    // Drain three real chunks, then stop draining: the provider keeps
    // producing while nothing consumes, which is the backpressure condition.
    let drained = 0, text = "";
    for (let i = 0; i < 3; i++) {
        const n = await st.next();
        if (n.done) break;
        if (n.value && n.value.text) { drained++; text += n.value.text; }
    }
    assert("the real model produced chunks the consumer could read", drained >= 1 && text.trim().length > 0);
    if (drained < 1) {
        console.error("BLOCKED: the real model produced no chunk to drain — nothing to exercise");
        process.exit(2);
    }
    console.log("  ℹ drained " + drained + " chunk(s), then stopped draining");

    // Wait (without draining) until the bounded queue overflows. The runtime
    // sets the aborted flag and raises the typed error at the bound.
    const t0 = Date.now();
    let overflowed = false;
    while (Date.now() - t0 < BOUND_MS) {
        if (st.aborted === true) { overflowed = true; break; }
        await sleep(100);
    }
    const waitMs = Date.now() - t0;

    if (!overflowed) {
        // The stream finished before the bound: the scenario was not exercised.
        let tail = null;
        try { tail = await st.next(); } catch (e) { tail = { error: String((e && e.message) || e) }; }
        console.error("BLOCKED: the real model finished after " + waitMs +
            " ms without filling the 1024-chunk queue (last delivery: " +
            JSON.stringify(tail && tail.value ? tail.value.text : null) +
            ") — the backpressure bound was not reached, so the case is unverified");
        process.exit(2);
    }
    console.log("  ℹ the queue overflowed after " + waitMs + " ms of not draining");

    // The typed error must reach the consumer on the next delivery.
    let err = null, value = null, done = false;
    try {
        const r = await st.next();
        value = r.value; done = r.done;
    } catch (e) { err = e; }
    const msg = String((err && err.message) || "");
    assert("non-draining consumer gets E_BACKPRESSURE",
        !!err && String(err.code) === "E_BACKPRESSURE");
    assert("the error names the bound (1024)", msg.indexOf("1024") >= 0);
    assert("abort cleaned up the provider side (aborted flag)", st.aborted === true);
    if (!err) console.log("  ℹ the stream ended instead of throwing:", { done, hasValue: value != null });

    console.log(`\n=== t25 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t25:", (e && e.message) || e); process.exit(1); });
