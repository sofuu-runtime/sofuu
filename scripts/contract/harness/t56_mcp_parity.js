// t56_mcp_parity.js — R5.6: a real MCP tool read and a real built-in tool
// read produce the same event/trace envelope shape, and the real tool result
// reaches a real model request whose answer carries the file's marker.
//
// Nothing is substituted: the MCP server is the local stdio server in
// real_file_mcp_server.js reading a real temporary file, and the model is a
// configured real backend. A model that does not call the tool, or whose
// answer does not carry the marker, is a failed run — never evidence.
//
// A model backend is sampled: one run of the agent is one sample of the
// model's behaviour. Each agent run is therefore attempted up to ATTEMPTS
// times and the case reports how many attempts it needed; a model that never
// complies is still a failed run.
//
// Usage (driven by scripts/test-real-agent-mcp.py):
//   sofuu run t56_mcp_parity.js <binary> <serverScript> <file> <receipt>
//       <starts> <marker> <provider> <model> <ctxWindow> [baseUrl]
// The bearer key of an OpenAI-wire endpoint arrives in
// SOFUU_TEST_PROVIDER_KEY, never in argv (argv is visible in `ps`).
'use strict';

const ATTEMPTS = 3;

async function main() {
  const argv = process.argv.slice(2);
  const [binary, serverScript, file, receipt, starts, marker, provider, model, windowText, baseUrl] = argv;
  const ctxWindow = Number(windowText);
  if (!binary || !serverScript || !file || !receipt || !starts || !marker ||
      !provider || !model || !Number.isSafeInteger(ctxWindow) || ctxWindow <= 0) {
    throw new Error('E_TEST_CONFIG');
  }
  const localEngine = provider === 'local';
  if (!localEngine && !baseUrl) {
    console.error('BLOCKED: a non-local provider needs baseUrl (an OpenAI-compatible endpoint)');
    process.exit(2);
  }
  if (localEngine && !/\.gguf$/i.test(model)) {
    console.error('BLOCKED: the local provider needs a .gguf model path');
    process.exit(2);
  }
  if (localEngine && !sofuu.fs.existsSync(model)) {
    console.error('BLOCKED: the local provider model path does not exist');
    process.exit(2);
  }
  // The file under test is the one the harness was given: it must hold the
  // fresh marker, so a stale sample can never satisfy the case.
  const sample = String(await sofuu.fs.readFile(file, 'utf8')).trim();
  if (sample !== marker) throw new Error('E_TEST_SAMPLE');

  const apiKey = String(process.env.SOFUU_TEST_PROVIDER_KEY || '');
  // Static protocol inputs exercise the pure normalizer, not a provider.
  const normalized = sofuu.tools.mcpResult({ content: [{type: 'text', text: 'sample'}], isError: false });
  const normalizedError = sofuu.tools.mcpResult({ content: [{type: 'text', text: 'failure'}], isError: true });
  if (!normalized.ok || normalized.text !== 'sample' || normalizedError.ok || normalizedError.text !== 'failure') {
    throw new Error('E_TEST_NORMALIZATION');
  }
  function def(name, system, tools) {
    const d = {
      name: name, provider, model,
      ctxWindow, max_tokens: 256, memory: 'off', rlm: 'off',
      budget: { maxSteps: 5, maxDepth: 0, maxWallMs: 120000 },
      system: system, tools: tools,
    };
    if (apiKey) d.api_key = apiKey;
    if (baseUrl) d.base_url = baseUrl;
    return d;
  }
  const MCP_SYSTEM = 'You have one tool: read_sample, which takes no parameters. Your first action must be a tool call to read_sample. Do not write any text before the call. After the tool returns, copy the exact contents it returned into your answer. Never guess the contents.';
  const BUILTIN_SYSTEM = 'You have one tool: read_file, which takes a path. Your first action must be a tool call to read_file with the path from the task. Do not write any text before the call. After the tool returns, copy the exact contents it returned into your answer. Never guess the contents.';

  // The built-in readers are jailed to the runtime's working directory, and
  // the jail compares paths lexically: a file handed over as an absolute
  // path outside that spelling (a symlinked temp dir, say) is refused. A
  // path relative to the working directory is therefore what the built-in
  // tool must be asked for.
  function toolPath(path) {
    try {
      const cwd = String(process.cwd() || '');
      if (cwd && path.indexOf(cwd + '/') === 0) return path.slice(cwd.length + 1);
    } catch (e) {}
    return path;
  }

  // The trace envelope fields a tool's events carry, independent of the
  // tool: the MCP tool and the built-in tool must emit the same shape.
  function envelopeShape(run, name) {
    const trace = Array.isArray(run.trace) ? run.trace : [];
    const shapes = [];
    for (const e of trace) {
      if ((e.kind === 'tool' || e.kind === 'tool_result') && e.payload && e.payload.name === name) {
        const key = e.kind + '|' + Object.keys(e).sort().join(',');
        if (shapes.indexOf(key) < 0) shapes.push(key);
      }
    }
    return shapes.sort().join(';');
  }

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
    // The tool really ran: a failed read is not a read.
    if (!results.every(e => e.payload.ok === true)) throw new Error('E_TEST_TOOL_FAILED');
    // The MCP tool and the built-in tool must agree on the envelope shape,
    // not merely on the answer.
    return { calls: calls.length, results: results.length };
  }
  // One attempt per sample of the model's behaviour; the case reports how
  // many it needed and never counts a non-compliant model as a pass. Each
  // attempt is an independent agent run under its own name.
  async function attempt(toolName, build, task) {
    let last = null;
    for (let i = 0; i < ATTEMPTS; i++) {
      const agent = sofuu.agent.define(build(i + 1));
      if (agent._def.ctxWindow !== ctxWindow) throw new Error('E_TEST_CONTEXT');
      const run = await sofuu.agent.run(agent, task);
      try {
        return { shape: checkEvents(run, toolName), attempts: i + 1, run: run };
      } catch (e) {
        last = e;
      }
    }
    throw last || new Error('E_TEST_AGENT_RESULT');
  }

  const mcp = await attempt('read_sample', (n) => def('real-mcp-check-' + n, MCP_SYSTEM,
    [{ mcp: [{ name: 'real-file-read', command: binary,
      args: ['run', serverScript, file, receipt, starts],
      tools: { allow: ['read_sample'] } }] }]),
    'Call read_sample and return the exact contents of the sample file.');
  // Independent receipt: the server wrote it only after the real read.
  const evidence = JSON.parse(await sofuu.fs.readFile(receipt, 'utf8'));
  if (evidence.operation !== 'read_sample' || evidence.contents !== marker) {
    throw new Error('E_TEST_MCP_RECEIPT');
  }
  // Independent receipt: the server process really started.
  const started = String(await sofuu.fs.readFile(starts, 'utf8')).trim();
  if (!started) throw new Error('E_TEST_MCP_START');

  const builtin = await attempt('read_file', (n) => def('real-builtin-check-' + n, BUILTIN_SYSTEM,
    ['read_file']),
    'Read ' + toolPath(file) + ' with read_file and report the exact contents it returns.');
  // The two tools must agree on the trace envelope shape. The number of
  // calls may differ (a model may retry); the envelope fields may not.
  if (envelopeShape(mcp.run, 'read_sample') !== envelopeShape(builtin.run, 'read_file')) {
    throw new Error('E_TEST_SHAPE_PARITY');
  }
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
  console.log('PASS real agent model and MCP file read (mcp attempts: ' + mcp.attempts +
    ', builtin attempts: ' + builtin.attempts + ')');
}

main().catch(function (error) {
  const code = String(error && (error.code || error.message) || 'E_TEST_FAILURE');
  console.error('FAIL real agent model and MCP file read:', /^E_[A-Z0-9_]+$/.test(code) ? code : 'E_TEST_FAILURE');
  process.exit(1);
});
