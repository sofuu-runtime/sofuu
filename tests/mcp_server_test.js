// tests/mcp_server_test.js — MCP stdio fixture, NOT a standalone test.
// P3-14 (AUDIT-2026-09-01): this file is a server program (it "passes"
// by starting, asserts nothing). Its job is being the in-repo spawn
// target for tests/mcp_client_test.js — which connects over stdio and
// hard-asserts echo/add behavior. Run the CLIENT test, not this file:
//   ./sofuu run tests/mcp_client_test.js
// Manual inspection (optional):
//   npx @modelcontextprotocol/inspector stdio ./sofuu run tests/mcp_server_test.js

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
