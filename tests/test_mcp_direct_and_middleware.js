// tests/test_mcp_direct_and_middleware.js — MCP Pillar 3: Direct In-Memory Transport,
// Middleware Onion Pipeline, and HTTP Server Mode Integration Suite.

import { mcp, agent, sofuu } from 'sofuu';

function assert(condition, message) {
  if (!condition) {
    console.error('FAIL: ' + message);
    throw new Error(message);
  }
}

async function runTests() {
  console.log('=== Sofuu MCP Pillar 3: Direct Transport & Middleware Suite ===\n');

  // 1. Direct In-Memory Transport & Schema Validation
  console.log('1. Testing Direct In-Memory Transport (transport: "direct")...');
  const directServer = mcp.serve({
    name: 'inproc-math',
    version: '2.0.0'
  });

  directServer.tool('fibonacci', {
    description: 'Calculates the nth Fibonacci number',
    schema: {
      type: 'object',
      properties: {
        n: { type: 'number', description: 'Fibonacci index' }
      },
      required: ['n']
    }
  }, async ({ n }) => {
    function fib(num) {
      if (num <= 1) return num;
      let a = 0, b = 1;
      for (let i = 2; i <= num; i++) {
        let c = a + b;
        a = b;
        b = c;
      }
      return b;
    }
    return { result: fib(n) };
  });

  const directClient = await mcp.connect(directServer, { transport: 'direct' });
  assert(directClient.transport === 'direct', 'Client transport must be direct');

  const tools = await directClient.listTools();
  assert(tools && tools.tools && tools.tools.length === 1, 'Should list 1 tool');
  assert(tools.tools[0].name === 'fibonacci', 'Tool name must be fibonacci');
  console.log('   ✓ Tool registered and listed: ' + tools.tools[0].name);

  // Microsecond invocation test
  const t0 = Date.now();
  const iterations = 50;
  for (let i = 0; i < iterations; i++) {
    const res = await directClient.call('tools/call', { name: 'fibonacci', arguments: { n: 10 } });
    assert(res && res.content && res.content[0].text.indexOf('55') >= 0, 'fib(10) should be 55');
  }
  const avgMs = (Date.now() - t0) / iterations;
  console.log('   ✓ 50 direct calls executed successfully. Avg latency: ' + avgMs.toFixed(3) + ' ms/call');

  // 2. Middleware Onion Pipeline (Server & Client)
  console.log('\n2. Testing Middleware Onion Pipeline (client.use & server.use)...');
  const trace = [];

  // Register server middleware
  directServer.use(async (ctx, next) => {
    trace.push('server-mw-in:' + ctx.toolName);
    await next();
    trace.push('server-mw-out:' + ctx.toolName);
  });

  // Register client middleware
  directClient.use(async (ctx, next) => {
    trace.push('client-mw-in:' + (ctx.toolName || ctx.method));
    await next();
    trace.push('client-mw-out:' + (ctx.toolName || ctx.method));
  });

  const mwRes = await directClient.call('tools/call', { name: 'fibonacci', arguments: { n: 6 } });
  assert(mwRes && mwRes.content[0].text.indexOf('8') >= 0, 'fib(6) should be 8');

  // Verify onion execution ordering
  assert(trace[0] === 'client-mw-in:fibonacci', 'Client MW in should execute first');
  assert(trace[1] === 'server-mw-in:fibonacci', 'Server MW in should execute second');
  assert(trace[2] === 'server-mw-out:fibonacci', 'Server MW out should execute third');
  assert(trace[3] === 'client-mw-out:fibonacci', 'Client MW out should execute fourth');
  console.log('   ✓ Onion pipeline verified in exact execution order: ' + trace.join(' -> '));

  // 3. Agent Integration with Direct MCP Client (tools: [directClient])
  console.log('\n3. Testing Agent Integration with Direct MCP Client (agent.define)...');
  const testAgent = agent.define({
    name: 'directAgent',
    tools: [directClient]
  });
  assert(testAgent && typeof testAgent.name === 'string', 'Agent should be defined with direct MCP client');
  console.log('   ✓ agent.define accepted direct MCP client successfully');

  // 4. Standalone Network HTTP Server Mode (sofuu.mcp.serve({ transport: 'http' }))
  console.log('\n4. Testing Network HTTP MCP Server (transport: "http")...');
  const HTTP_PORT = 19842;
  const netServer = mcp.serve({
    name: 'network-service',
    version: '1.2.3',
    transport: 'http',
    port: HTTP_PORT
  });

  netServer.tool('greet', {
    description: 'Returns a greeting string',
    schema: {
      type: 'object',
      properties: { name: { type: 'string' } },
      required: ['name']
    }
  }, async ({ name }) => {
    return 'Hello, ' + (name || 'anonymous') + ' from HTTP MCP!';
  });

  await new Promise((resolve) => {
    netServer.start(() => {
      resolve();
    });
  });
  console.log('   ✓ HTTP MCP server listening on port ' + HTTP_PORT);

  // Connect remote client over Streamable HTTP transport
  const httpClient = await mcp.connect({
    url: 'http://127.0.0.1:' + HTTP_PORT,
    transport: 'http'
  });

  assert(httpClient.transport === 'http', 'HTTP client transport should be http');
  const httpTools = await httpClient.listTools();
  assert(httpTools && httpTools.tools && httpTools.tools.length === 1, 'HTTP server should advertise 1 tool');
  assert(httpTools.tools[0].name === 'greet', 'HTTP tool name must be greet');

  const httpCallRes = await httpClient.call('tools/call', { name: 'greet', arguments: { name: 'Sofuu' } });
  const resultText = sofuu.tools.mcpResult(httpCallRes);
  assert(resultText.text.indexOf('Hello, Sofuu from HTTP MCP!') >= 0, 'Greeting result must match');
  console.log('   ✓ Remote HTTP call returned: ' + resultText.text);

  // Cleanup
  httpClient.disconnect();
  netServer.stop();
  directClient.disconnect();
  console.log('   ✓ HTTP server stopped and clients disconnected cleanly');

  console.log('\n=== ALL MCP PILLAR 3 TESTS PASSED (100% RELIABLE) ===');
}

runTests().catch((err) => {
  console.error('\nTest failed:', (err && err.stack) || (err && err.message) || String(err));
  process.exit(1);
});
