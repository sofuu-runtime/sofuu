// examples/rlm_mock_test.js — self-contained E2E test for sofuu.rlm.
//
// No network, no API keys, no Ollama: it starts an in-process
// OpenAI-compatible mock on 127.0.0.1 (sofuu.createServer), then runs a
// full sofuu.rlm.query against it. The mock implements a tiny scripted
// policy that drives the episode through:
//   round 1 — probe:      emit('probe', count()) + one direct chunk read
//   round 2 — recursion:  grep for the needle + llm() sub-call
//                         (the sub-prompt is a plain single-message
//                         completion answered by the same mock)
//   round 3 — final(answer)
//
// Asserts: the needle is in the final answer, ≥1 llm sub-call, ≥3 model
// rounds, and the trace contains llm + final events. Plus a few router
// matrix points. Exits non-zero on any failure.
//
// Run:  ./sofuu run examples/rlm_mock_test.js

const NEEDLE = "ZQ-NEEDLE-7734";

let failures = 0;
function check(name, cond) {
  if (cond) console.log("PASS " + name);
  else { failures++; console.log("FAIL " + name); }
}

/* The answer text the mock returns (OpenAI chat-completion shape). */
function replyJson(res, content) {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.send(JSON.stringify({
    choices: [{ message: { role: "assistant", content: content } }],
    usage: { prompt_tokens: 1, completion_tokens: 1 }
  }));
}

/* Wire fidelity: stream requests (the RLM driver's asks are stream-first
 * so Esc can kill them mid-flight) get SSE; non-stream gets plain JSON. */
function reply(res, body, content) {
  if (!(body && body.stream)) return replyJson(res, content);
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  const mid = Math.max(1, Math.ceil(content.length / 2));
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: content.slice(0, mid) } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ choices: [{ delta: { content: content.slice(mid) } }] }) + "\n\n");
  res.write("data: " + JSON.stringify({ usage: { prompt_tokens: 1, completion_tokens: 1 } }) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}

/* Slow drip: N chunks, one every perChunkMs — an ask against this branch
 * takes n*perChunkMs total, so a mid-flight abort finishes far earlier. */
let DRIPS_DONE = 0;
function sseDrip(res, n, perChunkMs) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  let i = 0;
  const t = setInterval(() => {
    i++;
    if (i > n) {
      clearInterval(t);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }
    DRIPS_DONE++;
    res.write("data: " + JSON.stringify({ choices: [{ delta: { content: "drip" + i + " " } }] }) + "\n\n");
  }, perChunkMs);
}

/* Minimal OpenAI-compatible POST handler with a scripted policy. */
function handler(req, res) {
  let body;
  try { body = JSON.parse(req.body || "{}"); }
  catch (e) { return replyJson(res, "mock: bad json"); }
  const msgs = body.messages || [];
  const last = String(msgs.length ? (msgs[msgs.length - 1].content || "") : "");
  const isMain = msgs.length > 0 && String(msgs[0].role) === "system";

  /* Mid-request abort probe: the kickoff question carries ABORT-DRIP —
   * the first driver ask lands here and drips slowly while the test sets
   * __rlm_aborted. */
  if (isMain && last.indexOf("ABORT-DRIP") >= 0)
    return sseDrip(res, 14, 150);

  /* Sub-prompt fulfillment: the driver sends plain single-user-message
   * completions (no RLM system prompt) for llm() calls. */
  if (!isMain) {
    sysLog("  mock: sub-prompt (" + last.length + " chars)");
    return reply(res, body, last.indexOf(NEEDLE) >= 0
      ? "sub-answer: the code word is " + NEEDLE
      : "sub-answer: nothing about a code word here");
  }

  if (last.indexOf("[snippet error") >= 0)
    return reply(res, body, "```js\nfinal(\"unexpected snippet error path\")\n```");

  /* Round 3: the llm sub-answer came back as [snippet result] → final. */
  if (last.indexOf("sub-answer: the code word is") >= 0)
    return reply(res, body, "```js\nfinal(\"RLM answer: the code word is " + NEEDLE + "\")\n```");

  /* Round 2: probe result is in → grep for the needle, recurse on its chunk. */
  if (last.indexOf("[snippet result]") >= 0)
    return reply(res, body,
      "```js\n" +
      "var hits = grep(\"" + NEEDLE + "\", 3);\n" +
      "var s = llm(\"needle-hunt: \" + chunk(hits[0].chunk));\n" +
      "\"llm-said: \" + s\n```");

  /* Round 1 (kickoff carries the question): probe the sandbox. */
  sysLog("  mock: main round (transcript " + msgs.length + " messages)");
  return reply(res, body,
    "```js\n" +
    "emit(\"probe\", \"chunks=\" + count());\n" +
    "var c0 = chunk(0).length;\n" +
    "\"probed \" + count() + \" chunks, c0=\" + c0\n```");
}

const VERBOSE = !!process.env.RLM_MOCK_VERBOSE;
function sysLog(s) { if (VERBOSE) console.log(s); }

/* ~120KB context, needle planted mid-corpus (never whole in one prompt). */
function buildContext() {
  const parts = [];
  for (let i = 1; i <= 110; i++) {
    parts.push("Memo " + i + "\n" +
      ("Routine filler for memo " + i + ": schedules, budgets, errands. ").repeat(14));
    if (i === 66) parts.push("NOTE: the code word is " + NEEDLE + " — keep it safe.");
  }
  return parts.join("\n\n");
}

async function main() {
  // (a) mock server on 127.0.0.1, first free of a few random-ish ports
  let server = null, port = 0;
  const base = 17000 + (Date.now() % 5000);
  for (let i = 0; i < 6; i++) {
    const p = base + i * 137;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, "127.0.0.1");
      port = p;
      break;
    } catch (e) { console.log("  port " + p + " failed: " + e); server = null; }
  }
  if (!server) { console.log("FAIL could not bind any mock port"); process.exit(1); }
  console.log("mock provider listening on 127.0.0.1:" + port);

  // sanity: sofuu.rlm was registered by the Rust seam
  check("sofuu.rlm exists", !!(sofuu.rlm && typeof sofuu.rlm.query === "function"));
  check("sofuu.rlm.VERSION", sofuu.rlm.VERSION === "0.1.0");

  // (c) full E2E query against the mock
  const context = buildContext();
  const res = await sofuu.rlm.query(context, "what is the code word?", {
    provider: "openai",
    base_url: "http://127.0.0.1:" + port + "/v1/chat/completions",
    api_key: "x",
    model: "mock",
    trace: true,
    maxWallMs: 60000
  });
  if (VERBOSE) console.log("result: " + JSON.stringify(res, null, 1).slice(0, 2000));

  check("answer contains the needle", String(res.answer || "").indexOf(NEEDLE) >= 0);
  check("calls >= 1 (llm sub-call fulfilled)", (res.calls || 0) >= 1);
  check("rounds >= 3", (res.rounds || 0) >= 3);
  check("no budget stop", !res.stopped);
  const kinds = (res.trace || []).map(e => e.kind);
  check("trace present", Array.isArray(res.trace) && res.trace.length > 0);
  check("trace has llm events", kinds.indexOf("llm") >= 0);
  check("trace has a final event", kinds.indexOf("final") >= 0);
  check("trace has chunk reads", kinds.indexOf("chunk") >= 0);
  check("depthReached is 0 (v1, no nesting)", res.depthReached === 0);

  // (d) router matrix through sofuu.rlm.route
  check("route: >80% of window → rlm", sofuu.rlm.route(900, 1000, "where is X defined") === "rlm");
  check("route: holistic mid-window → rlm", sofuu.rlm.route(500, 1000, "summarize all the errors") === "rlm");
  check("route: small context → plain", sofuu.rlm.route(100, 1000, "where is X defined") === "plain");
  check("route: mid-window specific → plain", sofuu.rlm.route(500, 1000, "where is X defined") === "plain");

  // (e) mid-request Esc abort: the kickoff ask drips 14 chunks × 150ms
  // (≈2.1s total). Setting __rlm_aborted at t≈400ms must kill the stream
  // MID-FLIGHT — the query resolves stopped:'aborted' far before the drip
  // would finish (the old between-rounds behavior consumed every chunk).
  {
    DRIPS_DONE = 0;
    const t0 = Date.now();
    setTimeout(() => { globalThis.__rlm_aborted = true; }, 400);
    const ab = await sofuu.rlm.query("some context", "ABORT-DRIP probe", {
      provider: "openai",
      base_url: "http://127.0.0.1:" + port + "/v1/chat/completions",
      api_key: "x",
      model: "mock",
      maxWallMs: 60000
    });
    const ms = Date.now() - t0;
    check("abort: stopped === 'aborted'", ab.stopped === "aborted");
    check("abort: killed mid-flight (" + ms + "ms « 2100ms drip, " +
          DRIPS_DONE + "/14 chunks)", ms < 1500 && DRIPS_DONE < 14);
    check("abort: flag cleared for the next episode",
      typeof sofuu.rlm.query === "function");
  }

  console.log(failures === 0
    ? "\nRLM MOCK TEST: ALL PASSED"
    : "\nRLM MOCK TEST: " + failures + " FAILURE(S)");
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => {
  console.error("FAIL exception: " + (e && e.stack || e));
  process.exit(1);
});
