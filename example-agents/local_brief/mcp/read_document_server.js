// A real, read-only stdio MCP service for the local_brief example.
//
// This is an external MCP server, not an agent sidecar: the agent logic lives
// only in ../agent.kaze and reaches this service through Sofuu's MCP client.
// The service is bound to the single file its operator selected on the command
// line and returns that file's contents. It never writes and never connects to
// anything outside the local filesystem.
'use strict';

const file = process.argv[2];
if (!file) {
  console.error('the local_brief read service needs the path of one operator-selected file');
  process.exit(2);
}

const server = sofuu.mcp.serve({ name: 'local-brief-read', version: '1' });
server.tool('read_document', {
  description: 'Read the operator-selected local document and return its contents.',
  schema: { type: 'object', properties: {}, additionalProperties: false },
}, async function () {
  const contents = await sofuu.fs.readFile(file, 'utf8');
  return { contents: contents };
});
server.start();
