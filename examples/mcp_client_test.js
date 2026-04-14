// examples/mcp_client_test.js
// Test sofuu.mcp.connect() — connect to our own MCP server and call tools
//
// Usage: First need an MCP server running. This test spins up our own sofuu MCP server.
// Run: ./sofuu run examples/mcp_client_test.js

const { mcp } = sofuu;

console.log("=== MCP Client Test ===\n");

// Connect to our echo MCP server subprocess
mcp.connect("./sofuu run examples/mcp_server_test.js")
    .then(async (client) => {
        console.log("Connected to MCP server!\n");

        // List all available tools
        const tools = await client.listTools();
        console.log("Available tools:", JSON.stringify(tools, null, 2));

        // Call the echo tool
        const echoResult = await client.call("tools/call", {
            name: "echo",
            arguments: { message: "Hello from Sofuu MCP client!" }
        });
        console.log("\necho result:", JSON.stringify(echoResult, null, 2));

        // Call the add tool
        const addResult = await client.call("tools/call", {
            name: "add",
            arguments: { a: 40, b: 2 }
        });
        console.log("add result:", JSON.stringify(addResult, null, 2));

        client.disconnect();
        console.log("\n=== MCP Client Test PASSED ===");
    })
    .catch(err => {
        console.error("MCP Error:", err);
    });
