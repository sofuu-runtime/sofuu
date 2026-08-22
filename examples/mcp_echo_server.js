// examples/mcp_echo_server.js — a tiny REAL MCP tool server for tests.
//
// Spawned as a child process by examples/agent_test.js (A6/F4a routing):
//   sofuu run examples/mcp_echo_server.js <name>
//
// Registers `shared_tool` (present on every instance — name-clash bait)
// plus `<name>_tool` (unique per instance). Every tool answers with a
// string that identifies the serving instance, so the test can prove a
// call landed on the RIGHT server and never silently crossed over.

// For `sofuu run <script> <name>` the runtime's argv is
// [runtime, <flag?>, script, name] — the name is always the LAST arg.

const name = String(process.argv[process.argv.length - 1] || "srv").trim() || "srv";

const server = sofuu.mcp.serve();

server.tool("shared_tool", {
  description: "Exists on every echo instance (clash bait)",
  schema: { type: "object", properties: { msg: { type: "string" } } },
}, (args) => {
  return "echo from " + name + "/shared_tool: " + JSON.stringify(args || {});
});

server.tool(name + "_tool", {
  description: "Unique to instance " + name,
  schema: { type: "object", properties: { msg: { type: "string" } } },
}, (args) => {
  return "echo from " + name + "/" + name + "_tool: " + JSON.stringify(args || {});
});

server.start();
