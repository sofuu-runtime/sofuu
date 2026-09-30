// tests/mock_prompt_dedupe_server.js — scripted LLM endpoint for the
// prompt-dedupe E2E (session-2, AUDIT-2026-09-07). Serves
// OpenAI-compatible completions on 127.0.0.1:<port>, SSE when 200.
//
// Policy (first match wins):
//   wire body contains "reasoning_effort"  → HTTP 400 "does not support
//       reasoning_effort" (THINK-class rejection → the driver retries
//       the turn WITHOUT effort)
//   model "mock-p1"                        → HTTP 401 invalid api key
//       (CAPACITY-class → the driver fails over to the next chain entry)
//   anything else                          → SSE PLAIN-OK
// Every request logs one line — `MOCK-DEDUPE: model=<m> think=<yes|no>` —
// so the shell can assert exactly how many wire attempts each turn took.
// Run: ./sofuu run tests/mock_prompt_dedupe_server.js <port>

const PORT = parseInt(process.argv[process.argv.length - 1] || "18995", 10);

function sseText(res, text) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: text } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: {} }],
    usage: { prompt_tokens: 12, completion_tokens: 6 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}
function httpError(res, code, msg) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.send(JSON.stringify({ error: { message: msg, code: code } }));
}

const server = sofuu.http.createServer((req, res) => {
  const raw = String(req.body || "");
  let parsed = {};
  try { parsed = JSON.parse(raw || "{}"); } catch (e) {}
  const model = String(parsed.model || "unknown");
  const think = raw.indexOf('"reasoning_effort"') >= 0;
  console.log("MOCK-DEDUPE: model=" + model + " think=" + (think ? "yes" : "no"));
  if (think) {
    httpError(res, 400, "This model does not support reasoning_effort");
  } else if (model === "mock-p1") {
    httpError(res, 401, "invalid api key");
  } else {
    sseText(res, "PLAIN-OK");
  }
});
server.listen(PORT, "127.0.0.1");
console.log("MOCK-DEDUPE-READY " + PORT);
