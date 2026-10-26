// Build A — Sofuu In-Process Substrate (the §8 benchmark's same app).
// Streaming chat + 2 tools + persistent memory + resume after restart +
// local fallback + an OpenAI-compatible HTTP endpoint.
// ONE process, ONE file, zero npm dependencies, public runtime API only.
'use strict';

const MODEL = process.env.SOFUU_MODEL || 'local-model';       // model swap = one field
const PROVIDER = process.env.SOFUU_PROVIDER || 'local';
const BASE_URL = process.env.SOFUU_BASE_URL;                  // cloud when set; local/offline otherwise
const API_KEY = process.env.SOFUU_API_KEY || 'local';

// 1. Two tools, registered on the public belt.
sofuu.agent.define({
  name: 'bench-assistant',
  system: 'You are the §8 benchmark assistant. Be brief.',
  model: MODEL, provider: PROVIDER,
  ...(BASE_URL ? { base_url: BASE_URL } : {}),
  api_key: API_KEY,
  memory: 'on',                       // persistent memory (QTSQ or capability-gated)
  tools: [
    { name: 'read_note', description: 'Read a note from the project notes file',
      parameters: { type: 'object', properties: { path: { type: 'string' } } },
      execute: (a) => sofuu.fs.readFile(String(a.path || 'notes.md')) },
    { name: 'count_chars', description: 'Count characters in a text',
      parameters: { type: 'object', properties: { text: { type: 'string' } } },
      execute: (a) => String((a && a.text || '').length) },
  ],
  approve: 'never',                   // read-only tools; no gate by policy
});

// 2. The app: one function, reused by CLI and HTTP.
async function turn(sessionId, text) {
  const r = await sofuu.agent.run('bench-assistant', text, { sessionId: sessionId });
  // resume after restart: the §5.2 session document at the canonical location
  return { answer: r.answer, usage: r.usage, sessionId: sessionId };
}

// 3. The HTTP endpoint (OpenAI-compatible protocol via one call).
sofuu.serve.start({ port: Number(process.env.PORT || 8080), host: '127.0.0.1',
                    token: process.env.SOFUU_SERVE_TOKEN || 'bench-token',
                    model: process.env.SOFUU_MODEL_PATH || undefined });

// 4. CLI entry (the same process serves + turns).
if (typeof process !== 'undefined' && process.argv && process.argv[2] === 'turn') {
  turn(process.argv[3] || 's-bench', process.argv[4] || 'hello')
    .then((r) => { console.log(JSON.stringify(r)); process.exit(0); })
    .catch((e) => { console.error(String(e && e.message || e)); process.exit(1); });
}
module.exports = { turn };
