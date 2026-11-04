const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: false });

rl.on('line', (line) => {
  if (!line.trim()) return;
  try {
    const req = JSON.parse(line);
    if (req.method === 'initialize') {
      const resp = {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'mock-mcp', version: '1.0.0' }
        }
      };
      process.stdout.write(JSON.stringify(resp) + '\n');
    } else if (req.method === 'notifications/initialized') {
      // notification
    } else if (req.method === 'tools/list') {
      const resp = {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          tools: [
            {
              name: 'echo_tool',
              description: 'Echoes back argument',
              inputSchema: { type: 'object', properties: { text: { type: 'string' } } }
            }
          ]
        }
      };
      process.stdout.write(JSON.stringify(resp) + '\n');
    } else if (req.method === 'tools/call') {
      const text = (req.params && req.params.arguments && req.params.arguments.text) || '';
      const resp = {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          content: [{ type: 'text', text: 'Echo: ' + text }]
        }
      };
      process.stdout.write(JSON.stringify(resp) + '\n');
    }
  } catch (e) {
    // ignore
  }
});
