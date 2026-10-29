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

// Orphan detection must look at OUR child only: any unrelated `sleep` on the
// machine (a loaded runner, another agent's wait loop) would otherwise make
// the case fail. The marker has to live in the child's own command line, and
// it has to survive /bin/sh: `exec -a` is a bashism (Linux's /bin/sh is dash
// — "exec: -a: not found", exit 127, no child at all), and without `exec` the
// shell forks, so the tool's kill would only take the wrapper and leave the
// sleep orphaned. A unique sleep DURATION is portable, keeps the marked
// process the direct child (dash execs in place), and no other process on the
// box plausibly sleeps for exactly this many seconds.
const MARKER = "sleep " + (30000 + (Date.now() % 50000));
const CHILD_CMD = "exec " + MARKER;
function pgrepChild() {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "pgrep", args: ["-f", MARKER],
            onStdout: (c) => { out += String(c); },
            onExit: () => resolve(out.trim().split(/\s+/).filter(Boolean)),
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
                    function: { name: "bash", arguments: JSON.stringify({ command: CHILD_CMD }) },
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
    let settleRejected = null;
    const t00 = Date.now();
    const runP = sofuu.agent.run("chaost3", "run the sleep command", {
        signal: SIGNAL,
        // approve:'all' still routes through the HOST's callback — fail-closed
        // without one (agent.js refuses to execute on a missing decider).
        // The decider is a RUN opt, not a def field.
        onApproval: function () { return true; },
    });
    // Attach handlers immediately — an unhandled rejection stalls the loop.
    // A rejection after cancel is a legitimate terminal (the contract asks
    // for prompt termination, not success): record it and return, do NOT
    // rethrow into the settle race — that made the case fail whenever the
    // cancel produced a rejection instead of a resolved run.
    const settleWatch = runP.then(
        (r) => {
            if (settleResolve) { const f = settleResolve; settleResolve = null; f("resolved"); }
            return r;
        },
        (e) => {
            settleRejected = String((e && e.message) || e);
            console.error(`  ℹ️ run REJECTED at +${Date.now() - t00} ms: ${settleRejected}`);
            if (settleResolve) { const f = settleResolve; settleResolve = null; f("rejected"); }
            return null;
        },
    );
    let childLive = false;
    const t0 = Date.now();
    // Startup precondition, not the tested bound: a loaded runner can take a
    // few seconds to get the tool's child visible.
    while (Date.now() - t0 < 10000) {
        const live = await pgrepChild();
        if (live.length > 0) { childLive = true; break; }
        await new Promise((r) => setTimeout(r, 100));
    }
    assert("the tool's subprocess is live", childLive);

    const tCancel = Date.now();
    const cRet = sofuu.agent.cancel(SIGNAL);
    console.error(`  ℹ️ cancel(${SIGNAL}) → ${cRet} at +${tCancel - t00} ms; ACTIVE=${JSON.stringify(sofuu.agent.list())}`);
    assert("cancel accepted via the registered signal", cRet === true);

    // ── the run settles promptly after cancel ───────────────────────────
    // Load-aware (the plan's blocking-flip prerequisite): the 3 s bound is
    // nominal; a saturated runner delays the harness's own 100 ms ticks, so
    // count that excess and extend the bound the same way T2.1 does. A
    // genuinely stuck run still fails once the 5 s slack budget is spent.
    let settled = null;
    let slack = 0;
    let prev = tCancel;
    while (!settled && Date.now() - tCancel < 3000 + slack) {
        settled = await Promise.race([
            settleWatch.then(() => ({ how: "settled", wall: Date.now() - tCancel })),
            new Promise((r) => setTimeout(() => r(null), 100)),
        ]);
        const now = Date.now();
        if (!settled) slack = Math.min(5000, slack + Math.max(0, now - prev - 100));
        prev = now;
    }
    const race = settled || { how: "timeout", wall: Date.now() - tCancel };
    const how = settleRejected ? `rejected: ${settleRejected}` : race.how;
    assert(
        `run settled after cancel (${race.wall} ms, ${how}` +
        (slack > 0 ? `, incl. ${slack} ms scheduler slack` : "") + ")",
        race.how !== "timeout"
    );

    // ── no orphan: the marked child died with the cancel ────────────────
    // Reaping is asynchronous (worse on a loaded runner); poll a little
    // longer so the case asserts "no orphan", not "no reap latency".
    let orphans = [];
    const tOrphan = Date.now();
    while (Date.now() - tOrphan < 3000) {
        orphans = await pgrepChild();
        if (orphans.length === 0) break;
        await new Promise((r) => setTimeout(r, 100));
    }
    assert("no orphan sleep process after cancel", orphans.length === 0);

    console.log(`\n=== t23 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t23:", (e && e.message) || e); process.exit(1); });
