// tests/mock_llm_server.js — tiny scripted LLM endpoint for chat E2E
// harnesses (tests/chat_mention_e2e.sh). Serves OpenAI-compatible
// completions on 127.0.0.1:<port> (argv[2]), SSE when body.stream.
//
// Response policy (first match wins):
//   system prompt contains "AGENT=helper"      → HELPER-RAN-OK
//   prompt carries an unmatched-@file marker   → SAW-MISSING-MARKER
//   otherwise                                  → PLAIN-OK
//
// Run:  ./sofuu run tests/mock_llm_server.js 18699

const PORT = parseInt(process.argv[process.argv.length - 1] || "18699", 10);
let curPtk = 12; /* set per-request; kept in scope for sse/jsonReply */

// Token accounting: the mock reports its real request size (chars/4 of
// every message) instead of a fixed usage — a provider that always
// reports prompt_tokens=12 makes the chat footer meter useless in E2E.
function sse(res, text) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  const mid = Math.max(1, Math.ceil(text.length / 2));
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: text.slice(0, mid) } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: text.slice(mid) } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ usage: { prompt_tokens: curPtk, completion_tokens: Math.max(1, Math.ceil(text.length / 4)) } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}

function jsonReply(res, text) {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.send(JSON.stringify({
    choices: [{ message: { role: "assistant", content: text } }],
    usage: { prompt_tokens: curPtk, completion_tokens: Math.max(1, Math.ceil(text.length / 4)) },
  }));
}

const server = sofuu.http.createServer((req, res) => {
  // Model-catalog route for the /model picker E2E (chat_models_e2e.sh):
  // an OpenAI-style listing so wizardFetchModels has something to fetch.
  if (String(req.url || "").indexOf("/models") >= 0) {
    res.writeHead(200, { "Content-Type": "application/json" });
    /* live-alpha publishes real caps (detect-on-select: /model must report
     * the detected window); live-beta/gamma deliberately publish none, so
     * the "endpoint publishes no limits" branch has a fixture too. */
    res.send(JSON.stringify({ object: "list", data: [
      { id: "live-alpha", context_length: 1048576, max_completion_tokens: 65536 },
      { id: "live-beta" }, { id: "live-gamma" },
    ] }));
    return;
  }
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  let sys = "", all = "";
  for (let i = 0; i < msgs.length; i++) {
    const c = String(msgs[i].content || "");
    all += " " + c;
    if (String(msgs[i].role) === "system") sys += " " + c;
  }
  curPtk = Math.max(1, Math.ceil(all.length / 4));
  let reply;
  if (sys.indexOf("AGENT=helper") >= 0) reply = "HELPER-RAN-OK";
  else if (all.indexOf("(file not found or empty)") >= 0 ||
           all.indexOf("(error:") >= 0) reply = "SAW-MISSING-MARKER";
  else if (all.indexOf("FILE-CONTENT-CANARY-8321") >= 0) reply = "SAW-FILE-CONTENT";
  else if (all.indexOf("REMEMBER-CANARY") >= 0) reply = "RECALL-HIT-QZ77";
  else reply = "PLAIN-OK";
  console.log("MOCK-LLM: reply=" + reply + " ptk=" + curPtk + " msgs=" + msgs.length + " bytes=" + all.length + " sizes=" + msgs.map(function (m) { return (m.role || "?").slice(0, 4) + ":" + String(m.content || "").length; }).join(",") + " heads=" + msgs.map(function (m) { return String(m.content || "").slice(0, 100).replace(/\n/g, "\\n"); }).join(" |") + (all.indexOf("REMEMBER-CANARY") >= 0 ? " RECALL-HIT" : ""));
  if (parsed.stream) sse(res, reply); else jsonReply(res, reply);
});
/* Hold the server reference for the process lifetime — an unreferenced
 * server object is finalized (and its socket closed) by the GC. */
server.listen(PORT, "127.0.0.1");

console.log("MOCK-LLM-READY " + PORT);
