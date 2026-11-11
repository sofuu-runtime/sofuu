// A real stdio MCP server for the file-read integration check. It exposes one
// read-only operation restricted to the single path supplied by its operator.
// A receipt is written only after the file was actually read successfully.
'use strict';

const file = process.argv[2];
const receipt = process.argv[3];
const starts = process.argv[4];
if (!file || !receipt || !starts) {
  console.error('real-file MCP server requires file, receipt and startup log paths');
  process.exit(2);
}

const server = sofuu.mcp.serve({ name: 'real-file-read', version: '1' });
server.tool('read_sample', {
  description: 'Read the operator-selected temporary sample file.',
  schema: { type: 'object', properties: {}, additionalProperties: false },
}, async function () {
  const contents = await sofuu.fs.readFile(file, 'utf8');
  await sofuu.fs.writeFile(receipt, JSON.stringify({ operation: 'read_sample', contents: contents }));
  return { contents: contents };
});
// Controlled failure: perform a real temporary-file write, then close the
// server before sending a reply. The caller cannot know completion from the
// transport alone and must not repeat this potentially side-effecting call.
server.tool('record_and_close', {
  description: 'Append one receipt and close before replying.',
  schema: { type: 'object', properties: {}, additionalProperties: false },
}, async function () {
  await sofuu.fs.appendFile(receipt, 'recorded\n');
  process.exit(0);
});
server.tool('record_and_wait', {
  description: 'Append one receipt before waiting for a real deadline.',
  schema: { type: 'object', properties: {}, additionalProperties: false },
}, async function () {
  await sofuu.fs.appendFile(receipt, 'recorded\n');
  await new Promise(resolve => setTimeout(resolve, 1000));
  return { recorded: true };
});
// Record each actual process start independently of tool-call receipts.
sofuu.fs.appendFile(starts, 'started\n').then(function () { server.start(); })
  .catch(function (error) { console.error(String(error)); process.exit(1); });
