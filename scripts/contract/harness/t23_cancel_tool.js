// t23_cancel_tool.js — R2.3: a cancelled run kills the in-flight tool's
// subprocess — no orphan, prompt settle.
//
// The mock provider returns a bash tool_call for `sleep 30`; the harness
// waits until the child is live, cancels the run through its registered
// signal, and asserts the run settled and the child is gone.
//
// Run: ./sofuu run scripts/contract/harness/t23_cancel_tool.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 45 s"); process.exit(1); }, 45000);

const SIGNAL = "t23-cancel";

function pgrepSleep() {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "pgrep", args: ["-x", "sleep"],
            onStdout: (c) => { out += String(c); },
            onExit: () => resolve(out.trim()),
        });
    });
}

async function main() {
    // ── mock provider: turn 1 returns a bash tool_call, turn 2 an answer ──
    let server = null, port = 0;
    const base = 26300 + (Date.now() % 2000);
    const TOOL_CALL_FRAME = JSON.stringify({
        choices: [{
            delta: {
                tool_calls: [{
                    index: 0, id: "call1", type: "function",
                    function: { name: "bash", arguments: JSON.stringify({ command: "sleep 30" }) },
                }],
            },
            finish_reason: "tool_calls",
        }],
    });
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        let body = {};
        try { body = JSON.parse(req.body || "{}"); } catch (e) {}
        const sawToolResult = JSON.stringify(body.messages || []).indexOf("sleep 30") >= 0;
        if (sawToolResult) {
            res.write("data: " + JSON.stringify({
                choices: [{ delta: { content: "done" } }],
            }) + "\n\n");
        } else {
            res.write("data: " + TOOL_CALL_FRAME + "\n\n");
        }
        res.write("data: [DONE]\n\n");
        res.end();
    };
    for (let i = 0; i < 6; i++) {
        const p = base + i * 141;
        try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; }
        catch (e) { server = null; }
    }
    assert("mock provider bound", server != null);
    const URL = "http://127.0.0.1:" + port + "/v1/chat/completions";

    sofuu.agent.define({
        name: "chaost3", system: "Run the command.",
        model: "mock", base_url: URL, api_key: "x",
        tools: [{ builtin: "bash" }],
        approve: "all", // auto-approve: the harness cancels, not the gate
    });

    // ── start the run; cancel once the tool's child is live ────────────
    let settleInfo = null;
    let settleResolve = null;
    const t00 = Date.now();
    if ((await pgrepSleep()) !== "") console.error("  ⚠️ sleep was live BEFORE the run started");
    const runP = sofuu.agent.run("chaost3", "run the sleep command", {
        signal: SIGNAL,
        // approve:'all' still routes through the HOST's callback — fail-closed
        // without one (agent.js refuses to execute on a missing decider).
        // The decider is a RUN opt, not a def field.
        onApproval: function () { return true; },
    });
    // Attach handlers immediately — an unhandled rejection stalls the loop.
    const settleWatch = runP.then(
        (r) => {
            if (settleResolve) { const f = settleResolve; settleResolve = null; f("resolved"); }
            return r;
        },
        (e) => {
            console.error(`  ℹ️ run REJECTED at +${Date.now() - t00} ms: ${(e && e.message) || e}`);
            if (settleResolve) { const f = settleResolve; settleResolve = null; f("rejected"); }
            throw e;
        },
    );
    let childLive = false;
    const t0 = Date.now();
    while (Date.now() - t0 < 5000) {
        if ((await pgrepSleep()) !== "") { childLive = true; break; }
        await new Promise((r) => setTimeout(r, 100));
    }
    assert("the tool's subprocess is live", childLive);

    const tCancel = Date.now();
    const cRet = sofuu.agent.cancel(SIGNAL);
    console.error(`  ℹ️ cancel(${SIGNAL}) → ${cRet} at +${tCancel - t00} ms; ACTIVE=${JSON.stringify(sofuu.agent.list())}`);
    assert("cancel accepted via the registered signal", cRet === true);

    // ── the run settles promptly after cancel ───────────────────────────
    const race = await Promise.race([
        settleWatch.then(() => ({ how: "settled", wall: Date.now() - tCancel })),
        new Promise((r) => setTimeout(() => r({ how: "timeout", wall: Date.now() - tCancel }), 3000)),
    ]);
    assert(`run settled after cancel (${race.wall} ms, ${race.how})`, race.how !== "timeout");

    // ── no orphan: the sleep child died with the cancel ─────────────────
    let orphans = "";
    const tOrphan = Date.now();
    while (Date.now() - tOrphan < 1500) {
        orphans = await pgrepSleep();
        if (orphans === "") break;
        await new Promise((r) => setTimeout(r, 100));
    }
    assert("no orphan sleep process after cancel", orphans === "");

    console.log(`\n=== t23 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t23:", (e && e.message) || e); process.exit(1); });
