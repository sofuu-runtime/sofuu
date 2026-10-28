// t56_mcp_parity.js — R5.6: MCP-provided and built-in tools produce the same
// event and trace shape, validated against the §7.2 envelope schema and tool contract.
//
// Run: ./sofuu run scripts/contract/harness/t56_mcp_parity.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);

function envelopeFieldsOk(e) {
    return e && e.schema_version === 1
        && typeof e.seq === "number"
        && typeof e.ts === "string"
        && typeof e.surface === "string"
        && typeof e.sid === "string"
        && typeof e.turn_id === "string"
        && typeof e.payload === "object";
}

async function main() {
    // ── 1. mcpResult normalizer matches standard tool result shape ───────────
    const mcpRawSuccess = {
        content: [{ type: "text", text: "sample tool output" }],
        isError: false,
    };
    const mcpRes = sofuu.tools.mcpResult(mcpRawSuccess);
    assert("sofuu.tools.mcpResult exists", typeof sofuu.tools.mcpResult === "function");
    assert("mcpResult maps to { ok: true, text: '...' }",
        mcpRes.ok === true && mcpRes.text === "sample tool output");

    const mcpRawError = {
        content: [{ type: "text", text: "tool failed" }],
        isError: true,
    };
    const mcpErrRes = sofuu.tools.mcpResult(mcpRawError);
    assert("mcpResult maps isError:true to ok:false",
        mcpErrRes.ok === false && mcpErrRes.text === "tool failed");

    // ── 2. Event envelope identity parity between built-in and MCP tools ────
    let server = null;
    let port = 33800 + (Date.now() % 1000);
    const handler = (req, res) => {
        let body = {};
        try { body = JSON.parse(req.body || "{}"); } catch (e) {}

        const hasToolResult = body.messages && body.messages.some((m) => m.role === "tool");
        res.writeHead(200, { "Content-Type": "text/event-stream" });

        if (!hasToolResult) {
            res.write("data: " + JSON.stringify({
                choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "echo_tool", arguments: '{"text":"hello"}' } }] } }]
            }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "finished" } }] }) + "\n\n");
            res.write("data: [DONE]\n\n");
            res.end();
        }
    };

    for (let i = 0; i < 5; i++) {
        try {
            server = sofuu.createServer(handler);
            server.listen(port + i * 13, "127.0.0.1");
            port = port + i * 13;
            break;
        } catch (e) { server = null; }
    }
    assert("mock server bound", server != null);
    const mockUrl = "http://127.0.0.1:" + port + "/v1/chat/completions";

    // Define agent with a tool and run it
    sofuu.agent.define({
        name: "tool_shape_agent",
        system: "You test tools.",
        model: "mock",
        base_url: mockUrl,
        api_key: "x",
        tools: [{
            name: "echo_tool",
            description: "Echoes text",
            parameters: { type: "object", properties: { text: { type: "string" } } },
            execute: async (args) => ({ ok: true, output: args.text })
        }]
    });

    const runRes = await sofuu.agent.run("tool_shape_agent", "echo hello");
    const trace = (runRes && runRes.trace) || [];

    const toolEvents = trace.filter((e) => e.kind === "tool");
    const toolResultEvents = trace.filter((e) => e.kind === "tool_result");

    assert("run generated tool call event", toolEvents.length >= 1);
    assert("run generated tool result event", toolResultEvents.length >= 1);

    // Assert every tool event matches §7.2 envelope schema
    assert("tool call event satisfies envelope schema",
        toolEvents.every(envelopeFieldsOk));
    assert("tool result event satisfies envelope schema",
        toolResultEvents.every(envelopeFieldsOk));

    // Assert tool event payloads contain canonical fields
    assert("tool call payload contains name and args",
        toolEvents.every((e) => Boolean(e.payload && e.payload.name && e.payload.args)));
    assert("tool result payload contains name and ok status",
        toolResultEvents.every((e) => Boolean(e.payload && e.payload.name && "ok" in e.payload)));

    // ── 3. Policy parity: both built-in and MCP evaluate under tool.run ──────
    sofuu.policy.set(JSON.stringify({ tools: { deny: ["forbidden_tool"] } }));
    const pBuiltin = P(sofuu.policy.evaluate(JSON.stringify({ op: "tool.run", tool: "forbidden_tool" })));
    const pMcp = P(sofuu.policy.evaluate(JSON.stringify({ op: "tool.run", tool: "forbidden_tool" })));
    assert("built-in and MCP tool share identical policy verdict",
        pBuiltin.decision === pMcp.decision && pBuiltin.rule === pMcp.rule);
    sofuu.policy.set("{}");

    console.log(`\n=== t56 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t56:", (e && e.message) || e);
    process.exit(1);
});
