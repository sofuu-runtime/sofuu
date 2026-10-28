// t26_terminal_fuzz.js — R2.6: Failure-injection fuzz: one terminal event,
// seq strictly increasing, no gaps, across error and failure scenarios.
//
// Run: ./sofuu run scripts/contract/harness/t26_terminal_fuzz.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 35 s"); process.exit(1); }, 35000);

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

async function main() {
    let mockServer = null;
    let mockPort = 31250 + (Date.now() % 1200);

    const handler = (req, res) => {
        let body = {};
        try { body = JSON.parse(req.body || "{}"); } catch (e) {}

        const hasToolResult = body.messages && body.messages.some((m) => m.role === "tool");
        res.writeHead(200, { "Content-Type": "text/event-stream" });

        if (!hasToolResult) {
            res.write("data: " + JSON.stringify({
                choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "broken_tool", arguments: "{}" } }] } }]
            }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "recovered" } }] }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
        }
    };

    for (let i = 0; i < 5; i++) {
        try {
            mockServer = sofuu.createServer(handler);
            mockServer.listen(mockPort + i * 19, "127.0.0.1");
            mockPort = mockPort + i * 19;
            break;
        } catch (e) { mockServer = null; }
    }
    assert("mock server bound", mockServer != null);
    const mockUrl = "http://127.0.0.1:" + mockPort + "/v1/chat/completions";

    // ── Scenario 1: Agent tool execution failure ────────────────────────────
    sofuu.agent.define({
        name: "fuzz_tool_agent",
        system: "You handle tool failures.",
        model: "mock",
        base_url: mockUrl,
        api_key: "x",
        tools: [{
            name: "broken_tool",
            description: "A tool that throws an error",
            parameters: { type: "object", properties: {} },
            execute: async () => { throw new Error("Injected tool failure!"); }
        }]
    });

    const r1 = await sofuu.agent.run("fuzz_tool_agent", "call failing_tool");
    checkTrace("Scenario 1 (agent tool failure)", (r1 && r1.trace) || []);

    // ── Scenario 2: Chat driver network failure path ─────────────────────────
    const chatEvents = [];
    try { sofuu.chat.init({}); } catch (e) {}
    try {
        await sofuu.chat.submit("trigger network error", {
            provider: "openai",
            model: "mock",
            api_key: "x",
            base_url: "http://127.0.0.1:9/v1/chat/completions", // unreachable dead port
            onEvent: (ev) => { if (ev && ev.kind) chatEvents.push(ev); },
        });
    } catch (e) {}

    checkTrace("Scenario 2 (chat network failure)", chatEvents);

    try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) {}

    console.log(`\n=== t26 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t26:", (e && e.message) || e);
    process.exit(1);
});
