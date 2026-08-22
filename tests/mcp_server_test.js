// examples/mcp_server_test.js
// Test sofuu.mcp.serve() — build a simple MCP tool server
// Run: ./sofuu run examples/mcp_server_test.js
// Then test with: npx @modelcontextprotocol/inspector stdio ./sofuu run examples/mcp_server_test.js

const { mcp } = sofuu;

const server = mcp.serve();

server.tool("echo", {
    description: "Echoes back the input message",
    schema: {
        type: "object",
        properties: {
            message: { type: "string", description: "The message to echo" }
        },
        required: ["message"]
    }
}, (args) => {
    return { echo: args.message, runtime: "sofuu" };
});

server.tool("add", {
    description: "Adds two numbers together",
    schema: {
        type: "object",
        properties: {
            a: { type: "number" },
            b: { type: "number" }
        },
        required: ["a", "b"]
    }
}, (args) => {
    return { result: args.a + args.b };
});

server.start();
