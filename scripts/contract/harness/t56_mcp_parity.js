// Real Sofuu agent test: an authorized MCP server reads an actual temporary
// file and a configured remote model receives its result. No Kaze sidecar.
'use strict';

async function main() {
  const [binary, serverScript, file, receipt, starts, marker, provider, model, baseUrl, windowText] = process.argv.slice(2);
  const ctxWindow = Number(windowText);
  if (!binary || !serverScript || !file || !receipt || !starts || !marker ||
      !provider || !model || !baseUrl || !Number.isSafeInteger(ctxWindow) || ctxWindow <= 0) {
    throw new Error('E_TEST_CONFIG');
  }
  // Static protocol inputs exercise the pure normalizer, not a provider.
  const normalized = sofuu.tools.mcpResult({ content: [{type: 'text', text: 'sample'}], isError: false });
  const normalizedError = sofuu.tools.mcpResult({ content: [{type: 'text', text: 'failure'}], isError: true });
  if (!normalized.ok || normalized.text !== 'sample' || normalizedError.ok || normalizedError.text !== 'failure') {
    throw new Error('E_TEST_NORMALIZATION');
  }
  const agent = sofuu.agent.define({
    name: 'real-mcp-check', provider, model, base_url: baseUrl,
    ctxWindow, max_tokens: 256, memory: 'off', rlm: 'off',
    budget: { maxSteps: 5, maxDepth: 0, maxWallMs: 120000 },
    system: 'Use the available tool to read the sample. Never guess its contents.',
    tools: [{ mcp: [{ name: 'real-file-read', command: binary,
      args: ['run', serverScript, file, receipt, starts],
      tools: { allow: ['read_sample'] } }] }],
  });
  if (agent._def.ctxWindow !== ctxWindow) throw new Error('E_TEST_CONTEXT');
  const result = await sofuu.agent.run(agent,
    'Call read_sample and return the exact contents of the sample file.');
  function checkEvents(run, name) {
    const trace = Array.isArray(run.trace) ? run.trace : [];
    const calls = trace.filter(e => e.kind === 'tool' && e.payload && e.payload.name === name);
    const results = trace.filter(e => e.kind === 'tool_result' && e.payload && e.payload.name === name);
    function envelope(e) {
      return e.schema_version === 1 && typeof e.seq === 'number' && typeof e.ts === 'string' &&
        typeof e.surface === 'string' && typeof e.sid === 'string' &&
        typeof e.turn_id === 'string' && e.payload && typeof e.payload === 'object';
    }
    if (!calls.length || !results.length || !calls.every(e => envelope(e) && e.payload.args) ||
        !results.every(e => envelope(e) && typeof e.payload.ok === 'boolean') ||
        !String(run.answer || '').includes(marker)) {
      throw new Error('E_TEST_AGENT_RESULT');
    }
  }
  checkEvents(result, 'read_sample');
  const evidence = JSON.parse(await sofuu.fs.readFile(receipt, 'utf8'));
  if (evidence.operation !== 'read_sample' || evidence.contents !== marker) {
    throw new Error('E_TEST_MCP_RECEIPT');
  }
  const builtin = sofuu.agent.define({
    name: 'real-builtin-check', provider, model, base_url: baseUrl,
    ctxWindow, max_tokens: 256, memory: 'off', rlm: 'off',
    budget: { maxSteps: 5, maxDepth: 0, maxWallMs: 120000 },
    system: 'Use the available tool to read the sample. Never guess its contents.',
    tools: ['read_file'],
  });
  if (builtin._def.ctxWindow !== ctxWindow) throw new Error('E_TEST_CONTEXT');
  const builtinResult = await sofuu.agent.run(builtin,
    'Use read_file to read ' + file + ' and return its exact contents.');
  checkEvents(builtinResult, 'read_file');
  // Denial is exercised against real configured tools after the successful
  // calls; a denied MCP read must not write its independent receipt.
  await sofuu.fs.rm(receipt);
  sofuu.policy.set(JSON.stringify({ tools: { deny: ['read_sample', 'read_file'] } }));
  const client = await sofuu.mcp.connect({command: binary, args: ['run', serverScript, file, receipt, starts]},
    {server: 'real-file-read'});
  try {
    let mcpDenied = null, builtinDenied = null;
    try { await client.call('read_sample', {}); } catch (e) { mcpDenied = e; }
    try { await sofuu.tools.execute('read_file', {path: file}); } catch (e) { builtinDenied = e; }
    if (!mcpDenied || mcpDenied.code !== 'E_POLICY_DENIED' || !builtinDenied ||
        builtinDenied.code !== 'E_POLICY_DENIED' || sofuu.fs.existsSync(receipt)) {
      throw new Error('E_TEST_POLICY');
    }
  } finally {
    client.disconnect();
    sofuu.policy.set('{}');
  }
  console.log('PASS real agent model and MCP file read');
}

main().catch(function (error) {
  const code = String(error && (error.code || error.message) || 'E_TEST_FAILURE');
  console.error('FAIL real agent model and MCP file read:', /^E_[A-Z0-9_]+$/.test(code) ? code : 'E_TEST_FAILURE');
  process.exit(1);
});
