// tests/mcp_client_test.js — MCP client test (assert-based, A0.2).
// Spawns the in-repo MCP fixture (tests/mcp_server_test.js) over stdio,
// lists its tools, and calls echo + add with hard value assertions.
// P2-26 (AUDIT-2026-09-01): the old catch block swallowed every failure
// as "Skipped" — a fully broken client passed as "API surface exists".
// Real failures now fail loudly.

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== MCP Client Test ===\n");

assert("sofuu.mcp is object", typeof sofuu.mcp === "object");
assert("sofuu.mcp.connect is function", typeof sofuu.mcp.connect === "function");

async function main() {
    const client = await sofuu.mcp.connect("./sofuu run tests/mcp_server_test.js");

    assert("client.listTools is function", typeof client.listTools === "function");

    const tools = await client.listTools();
    // listTools may return an array or an object with a tools property
    const toolList = Array.isArray(tools) ? tools : (tools.tools || []);
    assert("listTools returns array or object", Array.isArray(tools) || typeof tools === "object");
    assert("has echo tool", toolList.some(t => t.name === "echo"));
    assert("has add tool", toolList.some(t => t.name === "add"));

    const echoResult = await client.call("tools/call", {
        name: "echo",
        arguments: { message: "Hello from MCP client!" }
    });
    assert("echo result carries the echoed message",
        JSON.stringify(echoResult).indexOf("Hello from MCP client!") >= 0);
    assert("echo result names the runtime",
        JSON.stringify(echoResult).indexOf("sofuu") >= 0);

    const addResult = await client.call("tools/call", {
        name: "add",
        arguments: { a: 40, b: 2 }
    });
    assert("add(40, 2) === 42",
        JSON.stringify(addResult).indexOf("42") >= 0);

    client.disconnect();

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All MCP client tests PASSED");
}

main().catch(err => {
    console.error("FAIL MCP client test:", (err && err.message) || err);
    process.exit(1);
});
