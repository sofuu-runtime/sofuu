// tests/mock_fail_server.js — failure-mode LLM endpoint for debugging the
// chat "no response" issue. Mode is picked by a marker in the prompt:
//   MODE429       -> HTTP 429 + JSON error body (upstream rate limit shape)
//   MODEERRFRAME  -> HTTP 200 SSE whose only frame is an error object,
//                    the shape OpenRouter emits when the upstream provider
//                    fails AFTER headers were sent
//   MODEEMPTY     -> HTTP 200 SSE with [DONE] and zero content frames
//   MODESILENT    -> HTTP 200 headers, then never a byte (model went
//                    unresponsive — the ai.rs stall watchdog must fire)
//   MODELOOP      -> every request that carries tools gets another
//                    read_file tool call (burns planning rounds until the
//                    step budget breaches); the no-tool salvage round gets
//                    summary text. Pair with SOFUU_CHAT_MAX_STEPS=<n>.
//   MODECLOSE     -> transport cut: the server process exits BEFORE writing
//                    a response, so curl fails with a transport error
//                    ("Empty reply from server" / connection reset). The
//                    agent must classify it transient and retry; the retries
//                    then hit a dead port (a "Couldn't/Could not connect to server"
//                    transport error — libcurl wording varies by build), so
//                    the turn ends in a loud ✗ after 3 attempts instead of
//                    dying on the first cut. MUST be the LAST mode turn —
//                    it kills this mock process.
//   anything else -> normal SSE "PLAIN-OK"
// Run:  ./sofuu run tests/mock_fail_server.js 18790

const PORT = parseInt(process.argv[process.argv.length - 1] || "18790", 10);

function sse(res, frames) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  for (const f of frames) res.write("data: " + f + "\n\n");
  res.end();
}

const server = sofuu.http.createServer((req, res) => {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  /* Mode marker = LAST user message only, so chat history from earlier
   * turns can't re-trigger a mode. */
  let all = "";
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (String(msgs[i].role) === "user") { all = " " + String(msgs[i].content || ""); break; }
  }

  if (all.indexOf("MODE429") >= 0) {
    console.log("MOCK-FAIL: mode=429");
    res.writeHead(429, { "Content-Type": "application/json" });
    res.send(JSON.stringify({
      error: { message: "Provider returned error", code: 429,
               metadata: { raw: "mock-model is temporarily rate-limited upstream.",
                           limit_source: "upstream_provider_shared_pool" } },
    }));
    return;
  }
  if (all.indexOf("MODEERRFRAME") >= 0) {
    console.log("MOCK-FAIL: mode=errframe (200 + in-stream error object)");
    sse(res, [
      JSON.stringify({ error: { message: "Provider returned error", code: 429,
                                metadata: { raw: "mock-model upstream failed mid-stream." } } }),
      "[DONE]",
    ]);
    return;
  }
  if (all.indexOf("MODEEMPTY") >= 0) {
    console.log("MOCK-FAIL: mode=empty (200 + [DONE] only)");
    sse(res, ["[DONE]"]);
    return;
  }
  if (all.indexOf("MODESILENT") >= 0) {
    console.log("MOCK-FAIL: mode=silent (200 headers, then never a byte)");
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    /* Never write, never end — the ai.rs stall watchdog must catch this. */
    return;
  }
  if (all.indexOf("MODECLOSE") >= 0) {
    console.log("MOCK-FAIL: mode=close (process exit before response bytes)");
    /* Kernel closes the socket with zero response bytes → curl transport
     * error; the agent's retry classifier must treat it as transient. */
    process.exit(0);
  }
  /* MODELOOP — detected across ALL user messages, not just the last one:
   * on the salvage round the last user message is the salvage note
   * ("Budget limit reached…"), not the original prompt. */
  let anyUser = "";
  for (let i = 0; i < msgs.length; i++) {
    if (String(msgs[i].role) === "user") anyUser += " " + String(msgs[i].content || "");
  }
  if (anyUser.indexOf("MODELOOP") >= 0) {
    if (!parsed.tools || !parsed.tools.length) {
      console.log("MOCK-FAIL: mode=loop (salvage round -> summary text)");
      sse(res, [
        JSON.stringify({ choices: [{ delta: { content: "SALVAGED-SUMMARY" } }] }),
        JSON.stringify({ usage: { prompt_tokens: 12, completion_tokens: 6 } }),
        "[DONE]",
      ]);
      return;
    }
    console.log("MOCK-FAIL: mode=loop (tool round)");
    sse(res, [
      JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: "call_loop",
        function: { name: "read_file", arguments: JSON.stringify({ path: "README.md" }) } }] } }] }),
      JSON.stringify({ usage: { prompt_tokens: 12, completion_tokens: 6 } }),
      "[DONE]",
    ]);
    return;
  }
  console.log("MOCK-FAIL: mode=ok");
  sse(res, [
    JSON.stringify({ choices: [{ delta: { content: "PLAIN-OK" } }] }),
    JSON.stringify({ usage: { prompt_tokens: 12, completion_tokens: 6 } }),
    "[DONE]",
  ]);
});
server.listen(PORT, "127.0.0.1");
console.log("MOCK-FAIL-READY " + PORT);
