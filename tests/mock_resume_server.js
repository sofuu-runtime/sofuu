// tests/mock_resume_server.js — scripted LLM endpoint for the structured
// resume E2E (tests/chat_resume_e2e.sh). Serves OpenAI-compatible
// completions on 127.0.0.1:<port>, SSE when body.stream.
//
// Policy (first match wins):
//   resume block + tool canary in ANY message → RESUME-CONTEXT-OK
//   any tool-role message (a tool result this turn)            → TOOL-RAN-OK
//   task marker USE-TOOL-PLEASE                                → read_file tool call
//   otherwise                                                  → PLAIN-OK
//
// Run: ./sofuu run tests/mock_resume_server.js <port>

const PORT = parseInt(process.argv[process.argv.length - 1] || "18711", 10);

function sseToolCall(res, name, args) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write("data: " + JSON.stringify({ choices: [{ delta: { tool_calls: [
    { index: 0, id: "call_r1", type: "function",
      function: { name: name, arguments: JSON.stringify(args) } } ] } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: {} }],
    usage: { prompt_tokens: 40, completion_tokens: 8 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}
function sseText(res, text) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: text } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: {} }],
    usage: { prompt_tokens: 40, completion_tokens: 8 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}

  const server = sofuu.http.createServer((req, res) => {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = body.messages || [];
  let all = "";
  let lastUser = -1;
  let lastUserText = "";
  let lastTool = -1;
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    all += " " + String(m.content || "");
    if (String(m.role) === "user") { lastUser = i; lastUserText = String(m.content || ""); }
    if (String(m.role) === "tool") lastTool = i;
  }
  /* Claude-style retention keeps prior turns' tool transcripts in the
   * history, so "a tool message exists" no longer means "a tool ran
   * this turn" — only one AFTER the current user message does. */
  const hasToolResult = lastTool > lastUser;
  if (all.indexOf("tool activity from the turns above") >= 0 &&
      all.indexOf("RESUME-TOOL-CANARY-9271") >= 0) {
    console.log("MOCK-RESUME: resume context reached the model");
    sseText(res, "RESUME-CONTEXT-OK");
  } else if (hasToolResult) {
    console.log("MOCK-RESUME: tool result present; roles=" +
                msgs.map(m => m.role).join(","));
    sseText(res, "TOOL-RAN-OK");
  } else if (lastUserText.indexOf("USE-TOOL-PLEASE") >= 0) {
    sseToolCall(res, "read_file", { path: "notes.txt" });
  } else {
    sseText(res, "PLAIN-OK");
  }
});
server.listen(PORT, "127.0.0.1");
console.log("MOCK-RESUME-READY " + PORT);
