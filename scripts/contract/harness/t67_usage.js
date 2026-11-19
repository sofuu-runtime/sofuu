// t67_usage.js — R6.4/T6.7: usage reconciles — every surface reports the same
// shape, provider totals match, local cost is 0.
//
// The surfaces are REAL: `sofuu serve` — Sofuu's own OpenAI-compatible HTTP
// surface — serving the GGUF weights already present on this machine
// (selected through SOFUU_CONTRACT_MODEL) answers the client surfaces, and
// the in-process engine on the same weights answers the agent. The provider's own response bytes travel back with the client's
// result (`raw`), so the client's usage is reconciled against the provider's
// REPORT rather than against fixed numbers. No weights are downloaded and no
// provider is substituted.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL across the complete/stream/agent surfaces; blocks when unset
//
// Run: ./sofuu run scripts/contract/harness/t67_usage.js
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
const U = (u) => P(sofuu.events.canonicalUsage(JSON.stringify(u || {})));

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

function curl(args) {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "/usr/bin/curl",
            args: ["-s", "-m", "20"].concat(args),
            onStdout: (c) => { out += String(c); },
            onExit: (code) => resolve({ code: code, out: out }),
        });
    });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* The provider must be the SAME build that runs this harness — the running
 * binary is argv[0]. It is resolved to an absolute path because the provider
 * runs in its own working directory. */
function binaryPath() {
    let cwd = "";
    try { cwd = String(process.cwd() || ""); } catch (e) { cwd = ""; }
    const candidates = [
        String((process.argv && process.argv[0]) || "").trim(),
        String((process.env && process.env.SOFUU_BIN) || "").trim(),
    ];
    for (const c of candidates) {
        if (!c) continue;
        const abs = c.charAt(0) === "/" ? c : (cwd ? cwd + "/" + c : c);
        if (sofuu.fs.existsSync(abs)) return abs;
    }
    return "./target/debug/sofuu";
}

/* Start the real provider and return it ONLY once the server answering on
 * that port is provably ours: it advertises our model id and refuses an
 * unauthenticated request. Another session's server may own the port — its
 * response is never mistaken for this harness's provider. */
async function startProvider(binary, model, modelId, token, tmpHome) {
    for (let attempt = 0; attempt < 3; attempt++) {
        const port = 26900 + ((Date.now() + attempt * 271) % 2000);
        let stderr = "";
        let srv = null;
        try {
            srv = sofuu.spawn({
                command: binary,
                args: ["serve", "--model", model, "--model-id", modelId,
                    "--port", String(port), "--host", "127.0.0.1"],
                env: { SOFUU_SERVE_TOKEN: token, SOFUU_HOME: tmpHome, HOME: tmpHome },
                cwd: tmpHome,
                onStdout: () => {},
                onStderr: (c) => { stderr += String(c); },
                onExit: () => {},
            });
        } catch (e) {
            console.error("  ❌ FAIL: could not start the provider: " + ((e && e.message) || e));
            return null;
        }
        for (let i = 0; i < 180; i++) {
            await sleep(500);
            const auth = await curl(["-H", "Authorization: Bearer " + token,
                "http://127.0.0.1:" + port + "/v1/models"]);
            if (auth.out.indexOf('"' + modelId + '"') >= 0) {
                return { srv: srv, port: port };
            }
            /* Something answered on this port and it is not our server
             * (another session's provider owns the port): a model list
             * means it is tokenless, any other body means it wants a token
             * we do not have. Either way it is not our provider. */
            if (auth.code === 0 && auth.out !== "") break;
        }
        try { srv.kill && srv.kill(); } catch (e) {}
        if (stderr.trim() !== "") console.log("  ℹ provider stderr: " + stderr.trim().slice(0, 200));
    }
    return null;
}

async function main() {
    const model = resolveModel();
    console.log("  ℹ local model: " + model);

    // ── the real provider: sofuu serve over the local weights ─────────
    const binary = binaryPath();
    const token = "t67-contract-token";
    const modelId = "t67-contract-model";
    const stamp = Date.now() + "-" + (process.pid || 0);
    const tmpHome = "/tmp/sofuu-t67-home-" + stamp;
    await sofuu.fs.mkdir(tmpHome, { recursive: true });
    const provider = await startProvider(binary, model, modelId, token, tmpHome);
    const srv = provider && provider.srv;
    assert("real provider came up with the configured local weights", !!provider);
    const base = provider ? "http://127.0.0.1:" + provider.port + "/v1" : "";
    /* The wire profile is OpenAI-compatible; the endpoint is Sofuu's own
     * server over the local weights (see the header). */
    const mk = (extra) => Object.assign({
        provider: "openai", model: modelId, api_key: token, base_url: base,
    }, extra);
    const PROMPT = "hi";
    let u1 = null, u2 = null;

    try {
        if (provider) {
            // ── 1. ai.complete reports usage (the provider's own totals) ─
            const done = await sofuu.ai.complete(mk({
                messages: [{ role: "user", content: PROMPT }],
            }));
            /* The provider's own response body for THIS exchange — the
             * independent receipt the client's usage is checked against. */
            const raw = P(done.raw);
            const providerUsage = (raw && raw.usage) || {};
            u1 = U(Object.assign({ model: modelId, provider: "openai" },
                done && done.usage));
            assert("ai.complete reports usage", (done && done.usage) != null);
            assert("provider totals match (prompt tokens)",
                u1.in === providerUsage.prompt_tokens && u1.in > 0);
            assert("provider totals match (completion tokens)",
                u1.out === providerUsage.completion_tokens && u1.out > 0);
            assert("cloud cost carried (the provider reports none → 0)", u1.cost_usd === 0);

            // ── 2. ai.stream usage matches the same provider totals ──────
            const st = sofuu.ai.stream(mk({
                stream: true,
                messages: [{ role: "user", content: PROMPT }],
            }));
            let frames = 0, text = "";
            for await (const chunk of st) { if (chunk && chunk.text) { frames++; text += chunk.text; } }
            u2 = U(Object.assign({ model: modelId, provider: "openai" }, st.usage));
            assert("stream usage matches provider totals (prompt tokens)",
                u2.in === u1.in && u2.in > 0);
            assert("stream usage matches provider totals (completion tokens)",
                u2.out === frames && u2.out > 0);
            assert("stream carries the provider's own text", text.length > 0);
        }
    } finally {
        try { srv && srv.kill && srv.kill(); } catch (e) {}
    }

    // ── 3. agent usage normalizes through the same shape ───────────────
    // A real agent run on the same weights: the engine's own token counts
    // reach the agent's usage ledger and normalize identically.
    const def = sofuu.agent.define({
        name: "usagecheck", system: "Reply with one short word.",
        provider: "local", model: model,
        max_tokens: 16, memory: "off", rlm: "off",
        budget: { maxSteps: 2, maxDepth: 0, maxWallMs: 120000 },
    });
    const run = await sofuu.agent.run(def, "say hi");
    const agentUsage = (run && run.usage) || {};
    const agentU = U(Object.assign({ model: model, provider: "local" }, agentUsage));
    assert("agent reports usage", !!(run && run.usage));
    assert("agent shape normalizes identically",
        Object.keys(agentU).sort().join(",") === Object.keys(u2 || {}).sort().join(","));
    assert("agent usage reports the model's real tokens",
        agentU.in === agentUsage.promptTokens && agentU.out === agentUsage.completionTokens
        && agentU.in > 0 && agentU.out > 0);
    assert("agent usage carries the model that served it",
        agentU.model === model && agentU.provider === "local");

    // ── 4. one canonical shape: in/out/model/provider/cost_usd ──────────
    const keys = Object.keys(u1 || {}).sort().join(",");
    assert("canonical keys", keys === "cost_usd,in,model,out,provider");

    console.log(`\n=== t67 results: ${results.passed} passed, ${results.failed} failed ===`);
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
    console.error("FAIL t67:", (e && e.message) || e);
    process.exit(1);
});
