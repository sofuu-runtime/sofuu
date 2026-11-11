// Real MCP stdio test: the server reads a fresh file, records the read in a
// separate receipt, and propagates a real filesystem failure after removal.
'use strict';

async function main() {
  const file = process.argv[2];
  const receipt = process.argv[3];
  const marker = process.argv[4];
  const starts = process.argv[5];
  const serverScript = process.env.SOFUU_REAL_MCP_SERVER;
  const binary = process.env.SOFUU_BIN || process.argv[0];
  if (!file || !receipt || !marker || !starts || !serverScript) throw new Error('file, receipt, marker, startup log and server script required');
  if (sofuu.mcp._connectDirect || sofuu.mcp._connectHttp || sofuu.mcp._connectSse) {
    throw new Error('raw MCP transport constructors bypass the policy wrapper');
  }

  sofuu.policy.set(JSON.stringify({ tools: { deny: ['read_file'] } }));
  const readSpec = sofuu.tools.TOOLS.read_file;
  if (typeof readSpec.__exec === 'function') throw new Error('raw builtin executor exposes a policy bypass');
  const detachedRead = readSpec.execute;
  let tableDenied = null;
  try { await detachedRead({ path: file }); } catch (e) { tableDenied = e; }
  if (!tableDenied || tableDenied.code !== 'E_POLICY_DENIED') throw new Error('detached builtin executor bypassed policy');
  let builtinDenied = null;
  try { await sofuu.tools.execute('read_file', { path: file }); } catch (e) { builtinDenied = e; }
  if (!builtinDenied || builtinDenied.code !== 'E_POLICY_DENIED') throw new Error('builtin deny policy was bypassed');
  sofuu.policy.set(JSON.stringify({ approval: { mode: 'always' } }));
  let builtinApproval = null;
  try { await sofuu.tools.execute('read_file', { path: file }); } catch (e) { builtinApproval = e; }
  if (!builtinApproval || builtinApproval.code !== 'E_POLICY_APPROVAL_REQUIRED') throw new Error('builtin approval was bypassed');
  sofuu.policy.set('{}');
  const builtinRead = await sofuu.tools.execute('read_file', { path: file });
  if (String(builtinRead).indexOf(marker) < 0) throw new Error('builtin file read did not return the marker');

  const spec = {
    command: binary, args: ['run', serverScript, file, receipt, starts],
  };
  sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ['another-server'] } } }));
  let denied = false;
  try { await sofuu.mcp.connect(spec, { server: 'real-file-read' }); }
  catch (error) { denied = /denied by policy/i.test(String(error && error.message || error)); }
  if (!denied || sofuu.fs.existsSync(receipt) || sofuu.fs.existsSync(starts)) throw new Error('MCP connection denial did not prevent dispatch');
  sofuu.policy.set('{}');
  const client = await sofuu.mcp.connect(spec, { server: 'real-file-read' });
  try {
    const listing = await client.listTools();
    if (!listing || !Array.isArray(listing.tools) || !listing.tools.some(t => t.name === 'read_sample')) {
      throw new Error('real MCP server did not advertise read_sample');
    }
    sofuu.policy.set(JSON.stringify({ tools: { deny: ['read_sample'] } }));
    for (const args of [['tools/call', { name: 'read_sample', arguments: {} }], ['read_sample', {}]]) {
      let toolDenied = null;
      try { await client.call(args[0], args[1]); } catch (error) { toolDenied = error; }
      if (!toolDenied || toolDenied.code !== 'E_POLICY_DENIED' || sofuu.fs.existsSync(receipt)) {
        throw new Error('MCP tool policy denial did not prevent dispatch');
      }
    }
    sofuu.policy.set(JSON.stringify({ approval: { mode: 'always' } }));
    let approvalRequired = null;
    try { await client.call('read_sample', {}); } catch (error) { approvalRequired = error; }
    if (!approvalRequired || approvalRequired.code !== 'E_POLICY_APPROVAL_REQUIRED' || sofuu.fs.existsSync(receipt)) {
      throw new Error('MCP call without an approval receipt reached the server');
    }
    sofuu.policy.set('{}');
    const result = await client.call('tools/call', { name: 'read_sample', arguments: {} });
    const mapped = sofuu.tools.mcpResult(result);
    if (!mapped.ok || mapped.text.indexOf(marker) < 0) throw new Error('MCP result did not contain the file marker');
    const saved = JSON.parse(await sofuu.fs.readFile(receipt, 'utf8'));
    if (saved.operation !== 'read_sample' || saved.contents !== marker) throw new Error('independent file-read receipt did not match');

    await sofuu.fs.rm(file);
    await sofuu.fs.rm(receipt);
    const missing = await client.call('tools/call', { name: 'read_sample', arguments: {} });
    const failed = sofuu.tools.mcpResult(missing);
    if (failed.ok || sofuu.fs.existsSync(receipt)) throw new Error('missing file did not produce a real error');

    let uncertain = null;
    try { await client.call('tools/call', { name: 'record_and_close', arguments: {} }); }
    catch (error) { uncertain = error; }
    if (!uncertain || uncertain.code !== 'E_MCP_OUTCOME_UNKNOWN') {
      throw new Error('lost response did not report unknown completion');
    }
    // This second call is new work queued during reconnection. Its server also
    // records a real effect and exits before replying.
    if (client.state !== 'RECONNECTING') throw new Error('expected a reconnect in progress');
    let queuedUncertain = null;
    try { await client.call('record_and_close', {}); }
    catch (error) { queuedUncertain = error; }
    if (!queuedUncertain || queuedUncertain.code !== 'E_MCP_OUTCOME_UNKNOWN') {
      throw new Error('queued call with a lost response did not report unknown completion');
    }
    let queuedTimeout = null;
    try { await client.call('record_and_wait', {}, { timeoutMs: 50 }); }
    catch (error) { queuedTimeout = error; }
    if (!queuedTimeout || queuedTimeout.code !== 'E_TIMEOUT') {
      throw new Error('undispatched queued call did not time out');
    }
    // A queued call must be rechecked if policy changes before dispatch.
    const queuedDenied = client.call('record_and_wait', {}, { timeoutMs: 3000 });
    sofuu.policy.set(JSON.stringify({ tools: { deny: ['record_and_wait'] } }));
    let deniedWhileQueued = null;
    try { await queuedDenied; } catch (error) { deniedWhileQueued = error; }
    if (!deniedWhileQueued || deniedWhileQueued.code !== 'E_POLICY_DENIED') {
      throw new Error('changed policy did not deny the queued MCP call');
    }
    sofuu.policy.set('{}');
    // Allow the reconnect window to elapse: neither call may be replayed.
    await new Promise(resolve => setTimeout(resolve, 1500));
    let writes = await sofuu.fs.readFile(receipt, 'utf8');
    if (writes !== 'recorded\nrecorded\n') throw new Error('a side-effecting MCP call ran more than once');

    let timedOut = null;
    try { await client.call('record_and_wait', {}, { timeoutMs: 200 }); }
    catch (error) { timedOut = error; }
    if (!timedOut || timedOut.code !== 'E_MCP_OUTCOME_UNKNOWN') {
      throw new Error('timed-out call did not report unknown completion');
    }
    await new Promise(resolve => setTimeout(resolve, 1200));
    writes = await sofuu.fs.readFile(receipt, 'utf8');
    if (writes !== 'recorded\nrecorded\nrecorded\n') throw new Error('the timed-out call ran more than once');

    const startsBefore = await sofuu.fs.readFile(starts, 'utf8');
    try { await client.call('record_and_close', {}); } catch (error) {
      if (error.code !== 'E_MCP_OUTCOME_UNKNOWN') throw error;
    }
    sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ['another-server'] } } }));
    let reconnectDenied = null;
    try { await client.call('record_and_wait', {}, { timeoutMs: 3000 }); }
    catch (error) { reconnectDenied = error; }
    if (!reconnectDenied || reconnectDenied.code !== 'E_POLICY_DENIED') {
      throw new Error('changed server policy did not stop reconnection');
    }
    if (await sofuu.fs.readFile(starts, 'utf8') !== startsBefore) {
      throw new Error('a server started after reconnect policy denial');
    }
    sofuu.policy.set('{}');
    client.disconnect();
    await new Promise(resolve => setTimeout(resolve, 800));
    if (client.state !== 'CLOSED') throw new Error('disconnect was undone by a pending reconnect');
    console.log('PASS: real MCP file read, denial, missing-file error, deadlines, reconnect policy and no replay after lost responses');
  } finally {
    client.disconnect();
  }
}
main().catch(function (error) {
  console.error('FAIL real MCP file check:', String(error && error.message || error));
  process.exit(1);
});
