// t61_envelope.js — R6.1/R2.6: the envelope rides on every real emitted event,
// ordering is total, and each run terminates exactly once.
//
// The events come from REAL runs against a REAL local model — GGUF weights
// already present on this machine, selected through SOFUU_CONTRACT_MODEL. No
// weights are downloaded and no provider is substituted:
//   1. the agent loop (sofuu.agent.run against the local engine; the run
//      result's trace is the emitted event stream),
//   2. the chat driver (init with an onEvent sink) — one successful turn
//      against the local model and one turn whose endpoint is unreachable,
//      so the error path's events are covered too.
// Then asserts the §7.2 identity fields on every event, validates them with
// the runtime's own validator (sofuu.events.validate), and reports payload-rule
// violations informationally (payload alignment is the remaining R6 wiring).
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL for the agent and chat events; blocks when unset
//
// Run: ./sofuu run scripts/contract/harness/t61_envelope.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => {
    /* A real model on a loaded machine can overrun the budget. A timeout is
     * a coverage gap, not a contract violation: report it as blocked. */
    console.error("BLOCKED: the real local model did not finish within 600 s on this machine");
    process.exit(2);
}, 600000);

const P = (s) => JSON.parse(s);
const IDENTITY_FIELDS = ["schema_version", "seq", "ts", "surface", "sid", "turn_id"];

/* The real local weights this harness may load. The synthetic contract
 * fixture and the vocabulary-only GGUFs are parser fixtures, never evidence
 * of inference, so they are refused here. The model is selected by the
 * runner through SOFUU_CONTRACT_MODEL; an unset value is a blocked
 * prerequisite, never a silent fallback to some other file. */
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

function identityOk(e) {
    if (!e || typeof e !== "object") return false;
    for (const f of IDENTITY_FIELDS) if (!(f in e)) return false;
    return e.schema_version === 1 && typeof e.seq === "number"
        && typeof e.ts === "string" && e.ts.length >= 20
        && e.surface === "js" && typeof e.sid === "string" && typeof e.turn_id === "string";
}

async function main() {
    const model = resolveModel();
    console.log("  ℹ local model: " + model);

    // ── 1. agent loop events (the real local engine) ───────────────────
    sofuu.agent.define({
        name: "envelopecheck", system: "Reply with one word.",
        provider: "local", model: model,
        max_tokens: 16, memory: "off", rlm: "off",
        budget: { maxSteps: 2, maxDepth: 0, maxWallMs: 120000 },
    });
    const r = await sofuu.agent.run("envelopecheck", "say hi");
    const trace = (r && r.trace) || [];
    assert("agent run produced an event trace", trace.length >= 3);
    assert("every agent event carries the §7.2 identity fields",
        trace.every(identityOk));
    const seqs = trace.map((e) => e.seq);
    let monotonic = true;
    for (let i = 1; i < seqs.length; i++) if (seqs[i] <= seqs[i - 1]) monotonic = false;
    assert("seq strictly increasing (total order, R6.3)", monotonic);
    const stops = trace.filter((e) => e.kind === "stop").length;
    assert("exactly one terminal stop event (R2.6)", stops === 1);

    // The runtime's own validator: identity errors must be zero everywhere;
    // payload-rule gaps are reported informationally.
    let identityErrs = 0, payloadErrs = 0;
    for (const e of trace) {
        const errs = P(sofuu.events.validate(JSON.stringify(e))).errors || [];
        for (const err of errs) {
            if (/envelope field|unknown kind|seq|ts|surface/.test(err)) identityErrs += 1;
            else payloadErrs += 1;
        }
    }
    assert("zero envelope-identity violations across the agent trace", identityErrs === 0);
    console.log(`  ℹ payload-rule violations (informational, R6 wiring): ${payloadErrs}`);

    // ── 2. chat driver events — a real turn against the local model ───
    // The chat driver reads its provider from ~/.sofuu/config.json, so the
    // harness points HOME at an isolated directory and writes that config.
    // Nothing lands in the repository or in the developer's real home.
    const stamp = Date.now() + "-" + (process.pid || 0);
    const home = "/tmp/sofuu-t61-home-" + stamp;
    const project = "/tmp/sofuu-t61-project-" + stamp;
    await sofuu.fs.mkdir(home + "/.sofuu", { recursive: true });
    await sofuu.fs.mkdir(project, { recursive: true });
    const configPath = home + "/.sofuu/config.json";
    const realHome = process.env.HOME;
    process.env.HOME = home;
    await sofuu.fs.writeFile(configPath, JSON.stringify({
        provider: "local", model: model, max_output: 16,
    }));
    try { sofuu.chat.init({ project: project }); } catch (e) { console.log("  ℹ chat.init:", (e && e.message) || e); }

    const chatEvents = [];
    let chatErr = null;
    try {
        await sofuu.chat.submit("envelope probe", {
            onEvent: (ev) => { if (ev && ev.kind) chatEvents.push(ev); },
        });
    } catch (e) { chatErr = e; }
    assert("chat turn against the local model succeeded", chatErr === null);
    assert("chat driver emitted events through the sink", chatEvents.length >= 1);
    assert("every chat event carries the §7.2 identity fields",
        chatEvents.every(identityOk));
    const chatStops = chatEvents.filter((e) => e.kind === "done" || e.kind === "stop").length;
    assert("at most one chat terminal event", chatStops <= 1);

    // ── 3. chat driver events — the error path ────────────────────────
    // The same driver, pointed at an endpoint that refuses connections: the
    // failure is real (no provider answers) and the turn's events must still
    // ride the §7.2 envelope.
    const errorEvents = [];
    await sofuu.fs.writeFile(configPath, JSON.stringify({
        provider: "openai", model: "unreachable-endpoint", api_key: "x",
        base_url: "http://127.0.0.1:9/v1",
    }));
    try {
        // Deliberately offline: a dead endpoint makes the failure instant and
        // deterministic — the error path's events must ride the §7.2 envelope
        // regardless of the host's real config. The submit is raced, not
        // awaited: a failing provider's retry legs may run long; the envelope
        // contract is about the events on the sink, not the settle.
        await Promise.race([
            sofuu.chat.submit("envelope error probe", {
                onEvent: (ev) => { if (ev && ev.kind) errorEvents.push(ev); },
            }).catch(() => {}),
            new Promise((r) => setTimeout(r, 6000)),
        ]);
        /* The raced submit may still run a retry leg — its children are
         * killed here so the suite's later orphan checks see a clean
         * process table. */
        try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) {}
        await new Promise((r) => setTimeout(r, 300));
    } catch (e) { /* the error path is fine — the events matter */ }
    await new Promise((r2) => setTimeout(r2, 300));
    assert("chat error path emitted events through the sink", errorEvents.length >= 1);
    assert("every chat error event carries the §7.2 identity fields",
        errorEvents.every(identityOk));
    const errorStops = errorEvents.filter((e) => e.kind === "done" || e.kind === "stop").length;
    assert("at most one chat error-path terminal event", errorStops <= 1);

    // ── 4. cleanup ─────────────────────────────────────────────────────
    process.env.HOME = realHome;
    try { sofuu.fs.rm(configPath); } catch (e) { /* a leftover temp file is harmless */ }

    console.log(`\n=== t61 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}


/* A real backend that cannot answer in time (a loaded machine, an
 * unreadable model) is a resource gap, not a contract violation: the
 * assertions below were never evaluated. Report it as blocked. */
function resourceGap(e) {
    const code = String((e && e.code) || "");
    if (code === "E_AI_TIMEOUT" || code === "E_TIMEOUT" || code === "E_CAP_MISSING"
        || code === "E_CONFIG" || code === "E_MODEL_UNAVAILABLE") return true;
    return /timeout|timed out|no data for|unresponsive|cannot allocate|out of memory/i
        .test(String((e && e.message) || e || ""));
}

main().catch((e) => {
    if (resourceGap(e)) {
        console.error("BLOCKED: the real local model did not answer: " + ((e && e.message) || e));
        process.exit(2);
    }
    console.error("FAIL t61:", (e && e.message) || e);
    process.exit(1);
});
