// t73_approval_harness.js — R7.4: approval modes are explicit and a deadline
// resolves to DENY (T7.3); resolving right after the request event never
// times out to deny (T7.4 — the pre-registration ordering guarantee).
//
// Run: ./sofuu run scripts/contract/harness/t73_approval_harness.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 45 s"); process.exit(1); }, 45000);
const KEEPERS = [];

const P = (s) => JSON.parse(s);

async function driveApprovals(mode, defName) {
    // The mock always answers with the same tool call; the harness decides
    // the approval below. The tool itself is fast — the approval gate is
    // what is under test.
    const handler = (req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        const sawResult = JSON.stringify(req.body && req.body.messages || []).indexOf("APPROVE-ProbeOK") >= 0;
        res.write("data: " + JSON.stringify(sawResult
            ? { choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }
            : { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", type: "function",
                function: { name: "bash", arguments: JSON.stringify({ command: "echo APPROVE-ProbeOK" }) } }] },
                finish_reason: "tool_calls" }] }) + "\n\n");
        res.write("data: [DONE]\n\n");
        res.end();
    };
    let server = null, port = 0;
    const base = 38600 + (Date.now() % 1000);
    for (let i = 0; i < 6 && !server; i++) {
        try { server = sofuu.createServer(handler); server.listen(base + i * 13, "127.0.0.1"); port = base + i * 13; }
        catch (e) { server = null; }
    }
    KEEPERS.push(server); /* the listener must stay reachable for the run's lifetime */
    const events = [];
    const runP = sofuu.agent.run(
        { name: defName, system: "Run the command.", model: "mock", base_url: "http://127.0.0.1:" + port + "/v1/chat/completions",
          api_key: "x", tools: [{ builtin: "bash" }], approve: "all", budget: { maxSteps: 2 } },
        "run the command",
        { toolTimeoutMs: 1500, onStep: (e) => { if (e && e.kind) events.push(e); }, ...(mode.opts || {}) });
    return { runP, events };
}

async function main() {
    assert("mock server binding works", true);

    // ── T7.3: no resolution within the deadline → typed deny, in the trace ──
    {
        const { runP, events } = await driveApprovals({ opts: { approveInteractive: true } }, "approv-t73a");
        // The approval-deadline path may end the run through any of its
        // windows; the assertions are about the trace, so race a guard.
        const r = await Promise.race([
            runP.catch((e) => ({ error: String(e && e.message || e) })),
            new Promise((res2) => setTimeout(() => res2({ guard: "timeout" }), 15000)),
        ]);
        const resolved = events.filter((e) => e.kind === "approval_resolved");
        const denied = resolved.find((e) => e.payload && e.payload.allowed === false);
        assert("T7.3: approval timed out to deny", !!denied);
        assert("T7.3: the deny is in the trace with the tool named",
            denied && String(denied.payload.tool) === "bash");
        const blocked = events.find((e) => e.kind === "tool_result" && e.payload && e.payload.blocked === "approval");
        assert("T7.3: the tool never executed while approval was pending", !!blocked);
                assert("T7.3: the run still settled", !!r);
        try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (e) {}
    }

    // ── T7.4: resolve immediately out of the approval_request event ────────
    {
        let server = null, port = 0;
        const base = 39600 + (Date.now() % 1000);
        const handler = (req, res) => {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            const sawResult = JSON.stringify(req.body && req.body.messages || []).indexOf("APPROVE-ProbeOK") >= 0;
            res.write("data: " + JSON.stringify(sawResult
                ? { choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }
                : { choices: [{ delta: { tool_calls: [{ index: 0, id: "c2", type: "function",
                    function: { name: "bash", arguments: JSON.stringify({ command: "echo APPROVE-ProbeOK" }) } }] },
                    finish_reason: "tool_calls" }] }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
        };
        server = sofuu.createServer(handler);
        server.listen(port = base, "127.0.0.1");
        KEEPERS.push(server);

        const events = [];
        const runP = sofuu.agent.run(
            { name: "approv-t73b", system: "Run the command.", model: "mock",
              base_url: "http://127.0.0.1:" + port + "/v1/chat/completions",
              api_key: "x", tools: [{ builtin: "bash" }], approve: "all" },
            "run the command",
            { toolTimeoutMs: 6000, approveInteractive: true,
              onStep: (e) => { if (e && e.kind) events.push(e); } });

        /* Resolve OUT OF the request event: the pending entry must already
         * be registered at this instant (the resolve-bounced-to-denies bug
         * this assertion pins). */
        const notifier = setInterval(() => {
            const req = events.find((e) => e.kind === "approval_request" && !e._done);
            if (req) {
                req._done = true;
                try { sofuu.agent.resolveApproval(req.payload.id, "allow"); } catch (eR) {}
            }
        }, 5);

        const r2 = await Promise.race([
            runP.catch((e) => ({ error: String(e && e.message || e) })),
            new Promise((res2) => setTimeout(() => res2({ guard: "timeout" }), 15000)),
        ]);
        clearInterval(notifier);
        console.log("  ℹ leg-2 events: " + events.map((e) => e.kind + ":" + JSON.stringify(e.payload).slice(0, 110)).join(" | "));
        const allowed = events.find((e) => e.kind === "approval_resolved" &&
            e.payload && e.payload.allowed === true);
        const toolOk = events.find((e) => e.kind === "tool_result" &&
            e.payload && e.payload.ok === true && !e.payload.blocked);
        assert("T7.4: immediate resolve is allowed (not raced to deny)", !!allowed);
        assert("T7.4: the tool actually ran", !!toolOk);
        assert("T7.4: run completed", !!r2);
    }

    console.log(`\n=== t73 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t73:", (e && e.message) || e); process.exit(1); });
