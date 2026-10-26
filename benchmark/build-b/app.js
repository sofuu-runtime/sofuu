// Build B — the SAME §8 app on the fragmented stack (contrast Build A).
// Streaming chat + 2 tools + vector "memory" + resume + express endpoint.
const express = require('express');
const { Ollama } = require('@langchain/ollama');
const { ChromaClient } = require('chromadb');

const app = express();
app.use(express.json());
const llm = new Ollama({ baseUrl: 'http://ollama:11434', model: 'llama3.1:8b' });
const chroma = new ChromaClient({ path: 'http://chroma:8000' });
let collection; // async init, reconnect logic, health checks — all app code now

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!collection) collection = await chroma.getOrCreateCollection({ name: 'memory' });
    const docs = await collection.query({ queryTexts: [req.body.messages.at(-1).content], nResults: 3 });
    const stream = await llm.stream(String(req.body.messages.map((m) => m.content).join('\n')) +
      '\ncontext: ' + JSON.stringify(docs.documents));
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for await (const chunk of stream) res.write('data: ' + JSON.stringify({ delta: { content: chunk } }) + '\n\n');
    res.write('data: [DONE]\n\n'); res.end();
  } catch (e) { res.status(500).json({ error: String(e) }); }
});
// the two "tools", hand-rolled as routes — the agent loop, retries, timeouts,
// approvals, and the session store are now the app's problem:
app.post('/tools/read_note', async (req, res) => res.send(require('fs').readFileSync(req.body.path, 'utf8')));
app.post('/tools/count_chars', (req, res) => res.send(String(String(req.body.text || '').length)));
app.listen(8080);
