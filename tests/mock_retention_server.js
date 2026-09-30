// tests/mock_retention_server.js — scripted LLM endpoint for the
// Claude-style transcript-retention E2E (tests/chat_retention_e2e.sh).
// Serves OpenAI-compatible completions on 127.0.0.1:<port>, SSE when
// body.stream.
//
// Policy (first match wins):
//   any message contains NEEDLE-TT-88 (a tool result, mid-turn or
//     retained history)                        → RETENTION-SAW-NEEDLE
//   task marker USE-TOOL-TT                     → read_file tool call
//   otherwise                                   → PLAIN-OK
//
// Every request is logged with its per-message role summary so the e2e
// can assert that turn 2's request carries turn 1's tool messages.
//
// Run:  ./sofuu run tests/mock_retention_server.js <port>

const PORT = parseInt(process.argv[process.argv.length - 1] || "18731", 10);

function sseToolCall(res, name, args) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write("data: " + JSON.stringify({ choices: [{ delta: { tool_calls: [
    { index: 0, id: "call_tt1", type: "function",
      function: { name: name, arguments: JSON.stringify(args) } } ] } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: {} }],
    usage: { prompt_tokens: 60, completion_tokens: 8 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}
function sseText(res, text) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: text } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: {} }],
    usage: { prompt_tokens: 60, completion_tokens: 8 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}

const server = sofuu.http.createServer((req, res) => {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = body.messages || [];
  let all = "";
  let hasNeedle = false;
  let hasToolMsg = false;
  let lastUser = "";
  const roles = [];
  for (const m of msgs) {
    all += " " + String(m.content || "");
    if (String(m.role) === "tool") hasToolMsg = true;
    if (String(m.role) === "user") lastUser = String(m.content || "");
    if (m.tool_calls && m.tool_calls.length) roles.push("assistant+tc");
    else roles.push(String(m.role));
  }
  hasNeedle = all.indexOf("NEEDLE-TT-88") >= 0;
  console.log("MOCK-RET: roles=" + roles.join(",") +
              " needle=" + (hasNeedle ? "yes" : "no") +
              " toolmsg=" + (hasToolMsg ? "yes" : "no") +
              " last=" + JSON.stringify(lastUser.slice(0, 40)));
  if (hasNeedle) {
    sseText(res, "RETENTION-SAW-NEEDLE");
  } else if (lastUser.indexOf("USE-TOOL-TT") >= 0) {
    sseToolCall(res, "read_file", { path: "notes.txt" });
  } else {
    sseText(res, "PLAIN-OK");
  }
});
server.listen(PORT, "127.0.0.1");
console.log("MOCK-RET-READY " + PORT);
