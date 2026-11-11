// A real loopback MCP HTTP service reads and writes operator-provided files.
'use strict';

async function main() {
  const url = process.argv[2];
  const receipt = process.argv[3];
  const marker = process.argv[4];
  sofuu.policy.set('{}');
  const client = await sofuu.mcp.connect({ url: url, name: 'real-http-file', transport: 'http' });
  try {
    const ping = await client.call('ping', {}, { timeoutMs: 2000 });
    if (!ping || Object.keys(ping).length !== 0) throw new Error('HTTP ping did not reach the protocol method');
    sofuu.policy.set(JSON.stringify({ tools: { deny: ['read_sample'] } }));
    let denied = null;
    try { await client.call('read_sample', {}, { timeoutMs: 2000 }); } catch (e) { denied = e; }
    if (!denied || denied.code !== 'E_POLICY_DENIED' || sofuu.fs.existsSync(receipt)) {
      throw new Error('HTTP tool denial did not prevent dispatch');
    }
    sofuu.policy.set('{}');
    const read = sofuu.tools.mcpResult(await client.call('read_sample', {}, { timeoutMs: 2000 }));
    if (!read.ok || read.text.indexOf(marker) < 0) throw new Error('HTTP file result did not contain the marker');
    if ((await sofuu.fs.readFile(receipt, 'utf8')) !== marker) throw new Error('HTTP file receipt did not match');
    await sofuu.fs.rm(receipt);
    for (let i = 0; i < 2; i++) {
      let err = null;
      try { await client.call('record_and_wait', {}, { timeoutMs: 200 }); } catch (e) { err = e; }
      if (!err || err.code !== 'E_MCP_OUTCOME_UNKNOWN') throw new Error('HTTP deadline did not report unknown completion');
    }
    if (client.circuitState !== 'OPEN') throw new Error('HTTP timeouts did not trip the circuit breaker');
    let open = null;
    try { await client.call('ping', {}, { timeoutMs: 2000 }); } catch (e) { open = e; }
    if (!open || open.code !== 'E_MCP_CIRCUIT_OPEN') throw new Error('open circuit dispatched a request');
    await new Promise(resolve => setTimeout(resolve, 1200));
    const writes = await sofuu.fs.readFile(receipt, 'utf8');
    if (writes !== 'recorded\nrecorded\n') throw new Error('HTTP call was lost or replayed');
    console.log('PASS: real HTTP MCP file read, policy denial, ping, deadlines and circuit breaker');
  } finally {
    client.disconnect();
  }
}
main().catch(function (error) {
  console.error('FAIL real HTTP MCP check:', String(error && error.message || error));
  process.exit(1);
});
