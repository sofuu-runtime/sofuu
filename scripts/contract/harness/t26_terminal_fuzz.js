// t26_terminal_fuzz.js — R2.6: Failure-injection fuzz: one terminal event,
// seq strictly increasing, no gaps, across error and failure scenarios.
//
// Nothing is scripted: the agent leg runs a REAL local model (an existing
// real-weight GGUF, SOFUU_CONTRACT_MODEL) against the REAL built-in file
// tool, and the tool really fails — it reads a file that does not exist, so
// the failure is a real filesystem error, not an injected one. The chat leg
// drives the chat driver against an endpoint that refuses connections: a
// real network failure, and the turn ends with the runtime's own typed
// "completion status is unknown" error. No provider answers with a canned
// frame.
//
// A model that never asks for the tool leaves the agent leg unexercised: the
// case reports BLOCKED with the reason and never invents the event.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL with a real failing file read, plus a really refused connection; blocks when unset or when the model never calls the tool
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      ./sofuu run scripts/contract/harness/t26_terminal_fuzz.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 300 s"); process.exit(1); }, 300000);

const P = (s) => JSON.parse(s);
const ATTEMPTS = 3;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

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

function checkTrace(label, events) {
    assert(`${label}: emitted events (count=${events.length})`, events.length >= 1);

    // 1. Check monotonic seq with no gaps
    const seqs = events.map((e) => e.seq);
    let seqOk = true;
    for (let i = 1; i < seqs.length; i++) {
        if (seqs[i] !== seqs[i - 1] + 1) {
            seqOk = false;
            console.error(`  ❌ Gap in seq: events[${i - 1}].seq=${seqs[i - 1]}, events[${i}].seq=${seqs[i]}`);
            break;
        }
    }
    assert(`${label}: seq strictly increasing without gaps`, seqOk);

    // 2. Exactly one terminal event ('stop' or 'done')
    const terminals = events.filter((e) => e.kind === "stop" || e.kind === "done");
    assert(`${label}: exactly one terminal event (found ${terminals.length})`, terminals.length === 1);
}

/* One agent attempt: the real model is asked for the real built-in file
 * tool on a path that does not exist. The tool runs and really fails.
 * The task names a RELATIVE path on purpose: the built-in reader is jailed
 * to the working directory, so an absolute path would be refused by the
 * jail instead of reaching the filesystem. */
async function agentFailureAttempt(model, missing) {
    const name = "fuzz-tool-" + (Date.now() % 100000);
    sofuu.agent.define({
        name: name,
        system: "You have one tool: read_file, which takes a path. Your first action must be " +
                "a tool call to read_file with the path from the task. Do not write any text " +
                "before the call.",
        provider: "local", model: model,
        max_tokens: 128, memory: "off", rlm: "off",
        budget: { maxSteps: 2, maxDepth: 0, maxWallMs: 120000 },
        tools: ["read_file"],
    });
    let run = null, thrown = null;
    try {
        run = await sofuu.agent.run(name, "Read the file named " + missing +
            " in the current directory with read_file. Use the relative path " + missing + ".");
    } catch (e) { thrown = e; }
    try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) {}
    const trace = (run && run.trace) || [];
    return { trace: trace, thrown: thrown, missing: missing };
}

async function main() {
    const model = resolveModel();
    console.log("  ℹ model: " + model);

    // ── Scenario 1: Agent tool execution failure (real model, real tool) ──
    const missing = "t26-missing-" + (Date.now() % 1000000) + ".txt";
    let leg = null;
    for (let i = 0; i < ATTEMPTS && !leg; i++) {
        const attempt = await agentFailureAttempt(model, missing);
        const toolSeen = attempt.trace.some((e) => e.kind === "tool" && e.payload && e.payload.name === "read_file");
        if (toolSeen) { leg = attempt; break; }
        console.error("  ℹ attempt " + (i + 1) + "/" + ATTEMPTS +
            ": the real model did not ask for the tool — no tool event occurred");
    }
    if (!leg) {
        console.error("BLOCKED: the configured local model never asked for the real file tool in " +
            ATTEMPTS + " attempt(s), so the agent tool-failure trace was never produced");
        process.exit(2);
    }
    const toolResults = leg.trace.filter((e) => e.kind === "tool_result" && e.payload && e.payload.name === "read_file");
    assert("Scenario 1: the real tool really ran and really failed",
        toolResults.length >= 1 && toolResults.every((e) => e.payload.ok === false));
    assert("Scenario 1: the failure is a real filesystem error, not a policy refusal",
        toolResults.length >= 1 && !toolResults.some((e) => e.payload.blocked));
    checkTrace("Scenario 1 (agent tool failure)", leg.trace);

    // ── Scenario 2: Chat driver network failure path ─────────────────────
    // The driver reads its provider from its own configuration, so the
    // harness points HOME at an isolated directory and writes a config that
    // names an endpoint nothing listens on: the connection is really
    // refused. Nothing lands in the repository or in the real home.
    const stamp = Date.now() + "-" + (process.pid || 0);
    const home = "/tmp/sofuu-t26-home-" + stamp;
    const project = "/tmp/sofuu-t26-project-" + stamp;
    await sofuu.fs.mkdir(home + "/.sofuu", { recursive: true });
    await sofuu.fs.mkdir(project, { recursive: true });
    const configPath = home + "/.sofuu/config.json";
    const realHome = process.env.HOME;
    process.env.HOME = home;
    /* The provider ladder is shortened so the REAL failure surfaces inside
     * this case's window: "connection refused" is a transient error, and the
     * default ladder would ride it out for minutes. The failure itself is
     * unchanged — only how long the harness waits for it. */
    process.env.SOFUU_STREAM_RETRY_DELAYS = "300,300";
    await sofuu.fs.writeFile(configPath, JSON.stringify({
        provider: "openai", model: "unreachable-endpoint", api_key: "x",
        base_url: "http://127.0.0.1:9/v1",
    }));
    try { sofuu.chat.init({ project: project }); } catch (e) { console.log("  ℹ chat.init:", (e && e.message) || e); }

    const chatEvents = [];
    let chatErr = null;
    try {
        await Promise.race([
            sofuu.chat.submit("trigger network error", {
                onEvent: (ev) => { if (ev && ev.kind) chatEvents.push(ev); },
            }).catch((e) => { chatErr = e; }),
            wait(30000),
        ]);
        try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) {}
        await wait(300);
    } catch (e) { chatErr = e; /* the error path is fine — the events matter */ }
    await wait(300);
    checkTrace("Scenario 2 (chat network failure)", chatEvents);
    assert("Scenario 2: the failed turn really failed against the dead endpoint",
        !!chatErr && String(chatErr.code).length > 0);
    const chatStop = chatEvents.find((e) => e.kind === "stop" && e.payload && e.payload.reason === "error");
    assert("Scenario 2: the turn ends with the error terminal, not a completed answer",
        !!chatStop && !chatEvents.some((e) => e.kind === "done"));

    process.env.HOME = realHome;
    try { await sofuu.fs.rm(configPath); } catch (e) { /* a leftover temp file is harmless */ }

    console.log(`\n=== t26 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t26:", (e && e.message) || e); process.exit(1); });
