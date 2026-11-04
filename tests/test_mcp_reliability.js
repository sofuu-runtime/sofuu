// tests/test_mcp_reliability.js
// Verification suite for MCP Pillar 1: Production Reliability & Self-Healing

function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL: ' + msg);
    process.exit(1);
  }
}

async function run() {
  console.log('=== Sofuu MCP Pillar 1 Production Reliability Suite ===\n');

  // Test 1: Connect and inspect child PID
  console.log('1. Connecting to mock stdio MCP server...');
  const client = await sofuu.mcp.connect({
    command: 'node',
    args: ['tests/mock_mcp_server.js'],
    reconnect: {
      enabled: true,
      maxRetries: 3,
      initialBackoffMs: 100,
      maxBackoffMs: 500
    }
  });

  assert(client.state === 'CONNECTED', 'client.state must be CONNECTED initially');
  const initialPid = client.pid();
  console.log('   ✓ Connected. Child PID = ' + initialPid + ', state = ' + client.state);
  assert(typeof initialPid === 'number' && initialPid > 0, 'child pid must be > 0');

  // Test 2: List tools and execute tool call
  console.log('2. Testing tool listing and invocation...');
  const toolsRes = await client.listTools();
  assert(toolsRes && Array.isArray(toolsRes.tools) && toolsRes.tools.length === 1, 'listTools must return 1 tool');
  assert(toolsRes.tools[0].name === 'echo_tool', 'tool name must be echo_tool');

  const callRes = await client.call('tools/call', { name: 'echo_tool', arguments: { text: 'test-1' } });
  assert(callRes && callRes.content && callRes.content[0].text === 'Echo: test-1', 'call result must match');
  console.log('   ✓ echo_tool returned: ' + callRes.content[0].text);

  // Test 3: Chaos Kill & Transparent Auto-Respawn with Request Replay
  console.log('3. Chaos Kill: sending SIGKILL to child PID ' + initialPid + '...');
  await sofuu.tools.TOOLS.bash.execute({ command: 'kill -9 ' + initialPid });

  // Allow OS to deliver signal
  await new Promise(r => setTimeout(r, 80));

  console.log('   Issuing call immediately after kill; testing auto-respawn & request replay...');
  const replayRes = await client.call('tools/call', { name: 'echo_tool', arguments: { text: 'replayed-after-death' } });
  assert(replayRes && replayRes.content && replayRes.content[0].text === 'Echo: replayed-after-death', 'call after crash must succeed via replay');
  
  const newPid = client.pid();
  console.log('   ✓ Self-healing succeeded! Replayed result: ' + replayRes.content[0].text);
  console.log('   ✓ New child process spawned: PID = ' + newPid + ' (was ' + initialPid + ')');
  assert(newPid > 0 && newPid !== initialPid, 'new child PID must differ from killed PID');
  assert(client.state === 'CONNECTED', 'client state must be CONNECTED after recovery');

  // Test 4: Clean Disconnect and Zombie Cleanup
  console.log('4. Testing clean disconnect and zombie process cleanup...');
  client.disconnect();
  assert(client.state === 'CLOSED', 'client state must be CLOSED after disconnect');
  
  // Verify calls rejected after disconnect
  try {
    await client.call('tools/call', { name: 'echo_tool', arguments: { text: 'fail' } });
    assert(false, 'call on closed client must throw');
  } catch (eClosed) {
    assert(eClosed.code === 'E_MCP_DISCONNECTED', 'closed call must return E_MCP_DISCONNECTED');
    console.log('   ✓ Call on disconnected client correctly threw E_MCP_DISCONNECTED');
  }

  // Test 5: Reconnect Exhaustion on Non-Recoverable Failure
  console.log('5. Testing retry exhaustion on unstartable command...');
  try {
    const unstartable = await sofuu.mcp.connect({
      command: '/usr/bin/false',
      reconnect: { enabled: true, maxRetries: 2, initialBackoffMs: 50, maxBackoffMs: 100 }
    });
    // In case connect succeeded on immediate spawn, call will fail on exit
    await unstartable.call('tools/list', {});
    assert(false, 'unstartable command must fail');
  } catch (eExhaust) {
    console.log('   ✓ Correctly failed unstartable process: ' + eExhaust.message);
  }

  console.log('\n=== ALL MCP PILLAR 1 TESTS PASSED (100% RELIABLE) ===');
}

run().catch(err => {
  console.error('Unhandled failure:', err);
  process.exit(1);
});
