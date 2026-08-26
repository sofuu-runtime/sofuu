// examples/agent_test.js — self-contained E2E battery for sofuu.agent
// (PLAN-AGENTS A1–A5, A8, A9) + the web tools, over a scripted
// OpenAI-compatible mock provider (sofuu.createServer) and REAL MCP
// child servers (examples/mcp_echo_server.js).
//
// No network, no API keys. Covers:
//   A1  single agent + inline tool, result tree/usage/trace fields
//   A1  streamed tool-call fragments merge (delta.tool_calls split args)
//   A2  parent → sub-agent delegation (trace tree, answer merge)
//   A2  depth cap: child at maxDepth gets NO delegate tool (structurally)
//   A2  cycle guard: A→B→A surfaces a tool error, not a hang
//   A2  runMany concurrency cap (max in-flight ≤ configured)
//   A3  memory scopes shared/agent/off on one brain file (needs QTSQ;
//       auto-skips when sofuu.memory is unavailable)
//   A4  RLM fallback trace when sofuu.rlm fails; mapContext needle hunt;
//       rlm recurseVia routes sub-calls through a tool-using agent;
//       A4 FULL FORM — agent tools injected into the RLM sandbox
//       (tool() suspends → host executes via the agent tool path)
//   A5  budget_steps + budget_tokens stops; cancel mid-run; tool timeout
//   A6  MCP name→owner routing across two real servers + inline-wins clash
//   A8  loadDir (broken file listed, never fatal); list()
//   A9  renderTrace smoke
//   F4b Anthropic-wire stream-first tool calls (content_block tool_use +
//       input_json_delta fragments, usage input/output token mapping)
//   web built-in web_search tool end-to-end (via SOFUU_WEB_ENDPOINT mock)
//
// Run:  ./sofuu run examples/agent_test.js
// Verbose:  AGENT_TEST_VERBOSE=1 ./sofuu run examples/agent_test.js

let failures = 0;
let skips = 0;
function check(name, cond) {
  if (cond) { console.log("PASS " + name); }
  else { failures++; console.log("FAIL " + name); }
}
function skip(name, why) {
  skips++;
  console.log("SKIP " + name + " — " + why);
}
const VERBOSE = !!process.env.AGENT_TEST_VERBOSE;
function vlog(s) { if (VERBOSE) console.log(s); }

const RUNTIME = process.argv[0] || "./sofuu";

/* ── Mock provider (OpenAI-compatible; SSE when body.stream) ─────── */

const REQ_LOG = [];       // {start, end?} per request — concurrency probe
let REQ_N = 0;
let RLM_SYS = "";         // last RLM-driver system prompt (tool docs probe)
let LAST_OPENAI_RAW = ""; // P6: openai bodies must stay cache_control-free
let PARTIAL429_HITS = 0;  // retry-policy probes (A5): request counts
let EARLY429_HITS = 0;

function sysOf(msgs) {
  for (let i = 0; i < msgs.length; i++) {
    if (String(msgs[i].role) === "system") return String(msgs[i].content || "");
  }
  return "";
}
function toolCount(msgs) {
  let n = 0;
  for (let i = 0; i < msgs.length; i++) if (String(msgs[i].role) === "tool") n++;
  return n;
}
function allText(msgs) {
  let t = "";
  for (let i = 0; i < msgs.length; i++) t += " " + String(msgs[i].content || "");
  return t;
}

function sseOpen(res) { res.writeHead(200, { "Content-Type": "text/event-stream" }); }
function sseData(res, obj) { res.write("data: " + JSON.stringify(obj) + "\n\n"); }
function sseUsage(res) { sseData(res, { usage: { prompt_tokens: 40, completion_tokens: 8 } }); }
function sseEnd(res) { res.write("data: [DONE]\n\n"); res.end(); }

function sseText(res, text) {
  sseOpen(res);
  const mid = Math.max(1, Math.ceil(text.length / 2));
  sseData(res, { choices: [{ delta: { content: text.slice(0, mid) } }] });
  sseData(res, { choices: [{ delta: { content: text.slice(mid) } }] });
  sseUsage(res);
  sseEnd(res);
}
function sseToolCall(res, name, args, fragment) {
  sseOpen(res);
  const argsStr = JSON.stringify(args);
  if (fragment) {
    /* Split the arguments JSON across two delta fragments (OpenAI does
     * this in the wild — agent.js must merge by index). */
    const half = Math.ceil(argsStr.length / 2);
    sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_f1", function: { name: name, arguments: argsStr.slice(0, half) } }] } }] });
    sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: argsStr.slice(half) } }] } }] });
  } else {
    sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1", function: { name: name, arguments: argsStr } }] } }] });
  }
  sseUsage(res);
  sseEnd(res);
}

function decide(body) {
  const msgs = body.messages || [];
  const sys = sysOf(msgs);
  const tc = toolCount(msgs);
  const txt = allText(msgs);

  if (sys.indexOf("AGENT=simple") >= 0) {
    if (tc === 0) return { tool: { name: "get_weather", args: { city: "Paris" } } };
    return { text: "weather-final sees TOOLRES-9f3: " + (txt.indexOf("TOOLRES-9f3") >= 0 ? "yes" : "NO") };
  }
  if (sys.indexOf("AGENT=frag") >= 0) {
    if (tc === 0) return { tool: { name: "get_weather", args: { city: "Paris" }, fragment: true } };
    return { text: "frag-final sees TOOLRES-9f3: " + (txt.indexOf("TOOLRES-9f3") >= 0 ? "yes" : "NO") };
  }
  if (sys.indexOf("AGENT=orch") >= 0) {
    if (tc === 0) return { tool: { name: "delegate", args: { agent: "researcher", task: "find the code", why: "test" } } };
    return { text: "orch-final[" + (txt.indexOf("child-answer-42") >= 0 ? "has-child" : "no-child") + "]" };
  }
  if (sys.indexOf("AGENT=researcher") >= 0) {
    return { text: txt.indexOf("RECURSE-NEEDLE") >= 0 ? "RAN-VIA-AGENT-HIT" : "child-answer-42" };
  }
  if (sys.indexOf("AGENT=alpha") >= 0) {
    if (tc === 0) return { tool: { name: "delegate", args: { agent: "beta", task: "sub", why: "chain" } } };
    return { text: "alpha-done" };
  }
  if (sys.indexOf("AGENT=beta") >= 0) {
    if (tc === 0) return { tool: { name: "delegate", args: { agent: "alpha", task: "cycle attempt", why: "should fail" } } };
    return { text: txt.indexOf("cycle") >= 0 ? "beta-done-saw-cycle" : "beta-done" };
  }
  if (sys.indexOf("AGENT=capA") >= 0) {
    if (tc === 0) return { tool: { name: "delegate", args: { agent: "capB", task: "sub", why: "depth" } } };
    return { text: "capA-done" };
  }
  if (sys.indexOf("AGENT=capB") >= 0) {
    if (tc === 0) return { tool: { name: "delegate", args: { agent: "capA", task: "should not exist", why: "depth" } } };
    return { text: "capB-done" };
  }
  if (sys.indexOf("AGENT=loopy") >= 0) {
    /* A request WITHOUT tools is the breach salvage round (agent.js spends
     * one final no-tool call summarizing progress) — answer it. */
    if (!body.tools || !body.tools.length) return { text: "loopy-salvage-summary" };
    return { tool: { name: "get_weather", args: { city: "X" } } };
  }
  /* Retry-policy probes (A5): raw SSE frame sequences that don't fit the
   * {text}/{tool} verdict shape — the handler writes d.frames verbatim. */
  if (sys.indexOf("AGENT=partial429") >= 0) {
    PARTIAL429_HITS++;
    return { frames: [
      { choices: [{ delta: { content: "partial-text" } }] },
      { error: { message: "upstream died mid-stream", code: 429 } },
    ] };
  }
  if (sys.indexOf("AGENT=early429") >= 0) {
    EARLY429_HITS++;
    return { frames: [
      { error: { message: "upstream died before tokens", code: 429 } },
    ] };
  }
  if (sys.indexOf("AGENT=tooly") >= 0) {
    if (tc === 0) return { tool: { name: "slow_tool", args: {} } };
    return { text: txt.indexOf("timed out") >= 0 ? "tooly-done-saw-timeout" : "tooly-done" };
  }
  if (sys.indexOf("AGENT=chunkworker") >= 0) {
    return { text: txt.indexOf("MAP-NEEDLE-8891") >= 0 ? "needle-found-in-chunk" : "no-needle-here" };
  }
  if (sys.indexOf("AGENT=reducer") >= 0) {
    return { text: txt.indexOf("needle-found-in-chunk") >= 0 ? "merged-answer: MAP-NEEDLE-8891 present" : "merged-answer: absent" };
  }
  if (sys.indexOf("AGENT=webby") >= 0) {
    if (tc === 0) return { tool: { name: "web_search", args: { query: "rust vs zig" } } };
    return { text: "web-final[" + (txt.indexOf("MockResultTitle") >= 0 ? "ok" : "miss") + "]" };
  }
  if (sys.indexOf("AGENT=mcpt") >= 0) {
    if (tc === 0) return { tool: { name: "srv2_tool", args: { msg: "route me" } } };
    if (tc === 1) return { tool: { name: "shared_tool", args: { msg: "clash" } } };
    return { text: "mcp-final[" + (txt.indexOf("INLINE-TOOL-WINS") >= 0 ? "inline-ok" : "inline-miss") + "]" };
  }
  /* P4: answers <40 chars are skipped as trivial — the memory-scope
   * tests need a substantive answer to store. */
  if (sys.indexOf("AGENT=mem") >= 0) return { text: "MEM-XYZ-123 acknowledged and stored in agent memory for later recall." };
  if (sys.indexOf("AGENT=bigtool") >= 0) {
    if (tc === 0) return { tool: { name: "big_tool", args: {} } };
    return { text: "bigtool-final[" + (txt.indexOf("chars truncated") >= 0 ? "capped" : "raw") +
      (txt.indexOf("HEADMARKER") >= 0 ? "+head" : "-nohead") +
      (txt.indexOf("-TAILMARKER") >= 0 ? "+tail" : "-notail") + "]" };
  }
  if (sys.indexOf("AGENT=codetool") >= 0) {
    /* Built-in coding tools E2E: write → read → edit → grep → bash, then a
     * final answer that reflects exactly what each tool returned. */
    if (tc === 0) return { tool: { name: "write_file", args: { path: "src/app.js", content: "const A = 1;\nconst B = 2;\n" } } };
    if (tc === 1) return { tool: { name: "read_file", args: { path: "src/app.js" } } };
    if (tc === 2) return { tool: { name: "edit_file", args: { path: "src/app.js", old_string: "const B = 2;", new_string: "const B = 20;" } } };
    if (tc === 3) return { tool: { name: "grep", args: { pattern: "B = 20", path: "." } } };
    if (tc === 4) return { tool: { name: "bash", args: { command: "cat src/app.js" } } };
    return { text: "codetool-final[" +
      (txt.indexOf("created src/app.js") >= 0 ? "+write" : "-nowrite") +
      (txt.indexOf("2 lines") >= 0 ? "+read" : "-noread") +
      (txt.indexOf("1 replacement") >= 0 ? "+edit" : "-noedit") +
      (txt.indexOf("src/app.js:2:") >= 0 ? "+grep" : "-nogrep") +
      (txt.indexOf("[exit 0]") >= 0 && txt.indexOf("const B = 20;") >= 0 ? "+bash" : "-nobash") + "]" };
  }
  if (sys.indexOf("AGENT=coderail") >= 0) {
    /* Rails + error paths, asserted from the tool-error strings the loop
     * feeds back to the model: ambiguity, not-found, jail escape, and the
     * bash timeout kill. */
    if (tc === 0) return { tool: { name: "write_file", args: { path: "dup.txt", content: "same\nsame\n" } } };
    if (tc === 1) return { tool: { name: "edit_file", args: { path: "dup.txt", old_string: "same", new_string: "x" } } };
    if (tc === 2) return { tool: { name: "edit_file", args: { path: "src/app.js", old_string: "NO-SUCH-TEXT", new_string: "x" } } };
    if (tc === 3) return { tool: { name: "write_file", args: { path: "../escape.js", content: "x" } } };
    if (tc === 4) return { tool: { name: "bash", args: { command: "sleep 5", timeout_ms: 1200 } } };
    return { text: "coderail-final[" +
      (txt.indexOf("matches 2 places") >= 0 ? "+ambiguous" : "-noambiguous") +
      (txt.indexOf("not found") >= 0 ? "+notfound" : "-nonotfound") +
      (/escapes the project directory|jailed/.test(txt) ? "+jailed" : "-nojail") +
      (txt.indexOf("timed out") >= 0 ? "+timeout" : "-notimeout") + "]" };
  }
  if (sys.indexOf("AGENT=mlrep") >= 0) {
    /* ML gates: the model repeats the EXACT same call — the supervisor's
     * dup-call rule must attach a nudge to the second result. */
    if (tc === 0) return { tool: { name: "get_weather", args: { city: "Oslo" } } };
    if (tc === 1) return { tool: { name: "get_weather", args: { city: "Oslo" } } };
    return { text: "mlrep-final[" + (txt.indexOf("identical call already made") >= 0 ? "+nudge" : "-nonudge") + "]" };
  }
  if (sys.indexOf("AGENT=mlreread") >= 0) {
    /* ML gates: re-reading an unchanged file with DIFFERENT args (so the
     * dup-call rule doesn't fire first) — the re-read rule must fire. */
    if (tc === 0) return { tool: { name: "read_file", args: { path: "a.txt" } } };
    if (tc === 1) return { tool: { name: "read_file", args: { path: "a.txt", offset: 1 } } };
    return { text: "mlreread-final[" + (txt.indexOf("already read at step") >= 0 ? "+reread" : "-noread") + "]" };
  }
  if (sys.indexOf("AGENT=mlfreshclean") >= 0) {
    /* Control: fresh material must NOT produce a freshness notice.
     * (Checked BEFORE the mlfresh branch — its marker is a substring.) */
    if (tc === 0) return { tool: { name: "fetch_docs", args: { topic: "rivermesh" } } };
    return { text: "mlfreshclean-final[" + (txt.indexOf("[freshness]") >= 0 ? "+fresh" : "-clean") + "]" };
  }
  if (sys.indexOf("AGENT=mlfresh") >= 0) {
    /* ML freshness: the tool returns old-dated deprecated material — the
     * gate must put one evidence-carrying notice on the next context
     * boundary, which this mock reports having seen. */
    if (tc === 0) return { tool: { name: "fetch_docs", args: { topic: "zephyr" } } };
    return { text: "mlfresh-final[" + (txt.indexOf("[freshness]") >= 0 ? "+fresh" : "-nofresh") + "]" };
  }
  if (sys.indexOf("AGENT=mlrel") >= 0) {
    /* ML relevance: recalled memory mixes one on-task record with one
     * one-word wrong-topic trap — the pre-retrieval advisor must render a
     * [relevance] guidance notice on the ephemeral context message, which
     * this mock reports having seen. */
    return { text: "mlrel-final[" + (txt.indexOf("[relevance]") >= 0 ? "+rel" : "-norel") + "]" };
  }
  if (sys.indexOf("AGENT=slow") >= 0) {
    /* delayed response — drives the concurrency probe */
    return { text: "slow-done", delay: 350 };
  }

  /* No marker: the RLM driver's own calls (system-first) + single-user
   * plain sub-prompts. */
  if (msgs.length > 0 && String(msgs[0].role) === "system") {
    RLM_SYS = sys; /* probe: the RLM system prompt (tool docs land here) */
    /* A4 full form: an agent turn routed through RLM whose snippet calls
     * the sandbox tool() — the host executes it via the agent's tool path
     * and the snippet resumes with the result. */
    if (txt.indexOf("Question: use the sandbox tool") >= 0) {
      if (txt.indexOf("[snippet result]") >= 0) {
        return { text: '```js\nfinal("RLM-TOOL-OK")\n```' };
      }
      return { text: '```js\nvar r = tool("get_weather", { city: "Paris" }); "toolres:" + r\n```' };
    }
    if (txt.indexOf("[snippet error") >= 0) {
      const i = txt.indexOf("[snippet error");
      return { text: '```js\nfinal("SNIPPET-ERROR: ' + txt.slice(i + 15, i + 140).replace(/"/g, "'") + '")\n```' };
    }
    if (txt.indexOf("RAN-VIA-AGENT") >= 0 || txt.indexOf("[snippet result") >= 0) {
      return { text: '```js\nfinal("RLM-RECURSE-OK")\n```' };
    }
    return { text: '```js\nvar s = llm("needle check: " + chunk(0)); "saw " + s\n```' };
  }
  return { text: "sub-default" };
}

function handler(req, res) {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  LAST_OPENAI_RAW = req.body || "";
  const d = decide(body);
  REQ_N++;
  const entry = { n: REQ_N, start: Date.now(), end: 0 };
  REQ_LOG.push(entry);
  const finish = () => { entry.end = Date.now(); };
  vlog("  mock req #" + REQ_N + " stream=" + !!body.stream + " tools=" + (body.tools || []).length);
  if (d.frames) {
    /* Raw SSE frame sequences (e.g. in-stream error frames) — data frames
     * only; the terminating [DONE] is appended by sseEnd. */
    sseOpen(res);
    for (const f of d.frames) sseData(res, f);
    sseEnd(res);
    return;
  }
  if (d.delay) {
    setTimeout(() => { finish(); sseText(res, d.text); }, d.delay);
    return;
  }
  finish();
  /* Wire fidelity: stream requests get SSE; non-stream (ai.complete — the
   * RLM driver's calls) get a plain JSON completion body. */
  if (!body.stream) {
    res.writeHead(200, { "Content-Type": "application/json" });
    if (d.tool) {
      res.send(JSON.stringify({
        choices: [{ message: { role: "assistant", content: null,
          tool_calls: [{ id: "call_j1", type: "function",
            function: { name: d.tool.name, arguments: JSON.stringify(d.tool.args) } }] } }],
        usage: { prompt_tokens: 40, completion_tokens: 8 },
      }));
    } else {
      res.send(JSON.stringify({
        choices: [{ message: { role: "assistant", content: d.text } }],
        usage: { prompt_tokens: 40, completion_tokens: 8 },
      }));
    }
    return;
  }
  if (d.tool) sseToolCall(res, d.tool.name, d.tool.args, d.tool.fragment);
  else sseText(res, d.text);
}

/* ── Mock web endpoint (DDG-shaped HTML for the built-in tool) ───── */

function webHandler(req, res) {
  const html =
    '<div class="result">' +
    '<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frust&amp;rut=abc">MockResultTitle rust page</a>' +
    '<a class="result__snippet" href="//x">Mock snippet about rust vs zig</a>' +
    '</div>' +
    '<a class="result__a" href="https://example.com/second">MockResultTitle second</a>';
  res.writeHead(200, { "Content-Type": "text/html" });
  res.send(html);
}

/* ── Mock provider (Anthropic wire; SSE when body.stream) ──────────
 * Exercises F4b: content_block_start(tool_use) + input_json_delta
 * fragments must accumulate into stream.toolCalls so the agent loop
 * runs stream-first for Anthropic too. Usage rides message_start /
 * message_delta as {input_tokens, output_tokens}. */

let ANT_SYS = ""; // last-seen system prompt (tool docs probe)
let ANT_BODY = null;  // P6: last anthropic body (cache_control shape probe)
function antEvent(res, obj) { res.write("data: " + JSON.stringify(obj) + "\n\n"); }
function antToolStream(res, name, args) {
  const argsStr = JSON.stringify(args);
  const half = Math.ceil(argsStr.length / 2);
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  /* P6: cache accounting rides the same usage objects in the wild. */
  antEvent(res, { type: "message_start", message: { usage: { input_tokens: 25, output_tokens: 1,
    cache_creation_input_tokens: 333 } } });
  antEvent(res, { type: "content_block_start", index: 0, content_block: { type: "text" } });
  antEvent(res, { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Checking." } });
  antEvent(res, { type: "content_block_stop", index: 0 });
  antEvent(res, { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_01F4", name: name, input: {} } });
  antEvent(res, { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: argsStr.slice(0, half) } });
  antEvent(res, { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: argsStr.slice(half) } });
  antEvent(res, { type: "content_block_stop", index: 1 });
  antEvent(res, { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 9, cache_read_input_tokens: 777 } });
  antEvent(res, { type: "message_stop" });
  res.end();
}
function antTextStream(res, text) {
  const mid = Math.max(1, Math.ceil(text.length / 2));
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  antEvent(res, { type: "message_start", message: { usage: { input_tokens: 25, output_tokens: 1 } } });
  antEvent(res, { type: "content_block_start", index: 0, content_block: { type: "text" } });
  antEvent(res, { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: text.slice(0, mid) } });
  antEvent(res, { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: text.slice(mid) } });
  antEvent(res, { type: "content_block_stop", index: 0 });
  antEvent(res, { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 7 } });
  antEvent(res, { type: "message_stop" });
  res.end();
}
function anthropicHandler(req, res) {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  ANT_BODY = body;
  /* P6: the system now rides as a content-block array with a cache
   * marker — flatten it for the text probe. */
  if (Array.isArray(body.system)) {
    ANT_SYS = body.system.map(b => String((b && b.text) || "")).join("\n");
  } else if (typeof body.system === "string") {
    ANT_SYS = body.system;
  }
  const raw = JSON.stringify(body);
  if (raw.indexOf("tool_result") >= 0) {
    antTextStream(res, "anth-final[" + (raw.indexOf("ANTH-TOOLRES-77") >= 0 ? "ok" : "miss") + "]");
    return;
  }
  antToolStream(res, "get_weather", { city: "Paris" });
}

async function main() {
  /* sanity: the shipped drivers registered */
  check("sofuu.agent.define/run exist", !!(sofuu.agent && typeof sofuu.agent.define === "function" && typeof sofuu.agent.run === "function"));
  check("sofuu.agent.create (memory.rs) survived the merge", typeof sofuu.agent.create === "function");
  check("sofuu.web.search exists", !!(sofuu.web && typeof sofuu.web.search === "function"));

  /* mock servers */
  let server = null, port = 0;
  const base = 19000 + (Date.now() % 5000);
  for (let i = 0; i < 6; i++) {
    const p = base + i * 131;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, "127.0.0.1");
      let webPort = 0;
      /* web mock on its own port */
      for (let j = 0; j < 6; j++) {
        const wp = p + 500 + j * 17;
        try {
          const ws = sofuu.createServer(webHandler);
          ws.listen(wp, "127.0.0.1");
          webPort = wp;
          break;
        } catch (e) {}
      }
      if (!webPort) throw new Error("no web mock port");
      process.env.SOFUU_WEB_ENDPOINT = "http://127.0.0.1:" + webPort + "/";
      port = p;
      break;
    } catch (e) { server = null; }
  }
  if (!server) { console.log("FAIL could not bind mock ports"); process.exit(1); }
  const MOCK = "http://127.0.0.1:" + port + "/v1/chat/completions";
  const PROV = { provider: "openai", base_url: MOCK, api_key: "x", model: "mock" };
  /* defs need the provider embedded: delegation children and mapContext
   * workers do NOT inherit the parent's run opts. */
  const DEF_PROV = { provider: "openai", base_url: MOCK, api_key: "x", model: "mock" };
  vlog("mock provider on " + MOCK + ", web endpoint " + process.env.SOFUU_WEB_ENDPOINT);

  /* anthropic-wire mock on its own port */
  let antPort = 0;
  for (let i = 0; i < 6 && !antPort; i++) {
    const p = port + 900 + i * 19;
    try {
      const as = sofuu.createServer(anthropicHandler);
      as.listen(p, "127.0.0.1");
      antPort = p;
    } catch (e) {}
  }
  if (!antPort) { console.log("FAIL could not bind anthropic mock port"); process.exit(1); }
  const ANTH_MOCK = "http://127.0.0.1:" + antPort + "/v1/messages";
  vlog("anthropic mock on " + ANTH_MOCK);

  /* ── A1: single agent + inline tool ─────────────────────────────── */
  sofuu.agent.define({
    name: "simple",
    system: "AGENT=simple You report weather.",
    tools: [{
      name: "get_weather",
      description: "Get weather",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
      execute: async (args) => "TOOLRES-9f3 weather for " + (args && args.city) + ": 25C sunny",
    }],
    memory: "off",
    rlm: "off", provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("simple", "weather in Paris?", {});
    check("A1 answer flows tool result", r.answer.indexOf("weather-final sees TOOLRES-9f3: yes") >= 0);
    check("A1 usage populated", r.usage.llmCalls >= 2 && r.usage.toolCalls === 1 && r.usage.promptTokens > 0);
    check("A1 steps counted", r.steps === 1);
    check("A1 trace has tool + answer", (() => {
      const kinds = r.trace.map(e => e.kind);
      return kinds.indexOf("tool") >= 0 && kinds.indexOf("tool_result") >= 0 && kinds.indexOf("answer") >= 0;
    })());
    check("A1 not stopped", !r.stopped);
    check("A1 runId assigned", typeof r.runId === "string" && r.runId.length > 0);
  }

  /* A1: fragmented streamed tool-call args merge */
  sofuu.agent.define({
    name: "frag",
    system: "AGENT=frag You report weather.",
    tools: [{
      name: "get_weather", description: "Get weather",
      parameters: { type: "object", properties: { city: { type: "string" } } },
      execute: async (args) => "TOOLRES-9f3 weather " + JSON.stringify(args),
    }],
    memory: "off", rlm: "off", provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("frag", "weather in Paris?", {});
    check("A1 fragmented tool args merged", r.answer.indexOf("frag-final sees TOOLRES-9f3: yes") >= 0);
    check("A1 fragment carried city arg", r.trace.some(e => e.kind === "tool" && String(e.payload && e.payload.args || "").indexOf("Paris") >= 0));
  }

  /* ── A2: delegation ─────────────────────────────────────────────── */
  sofuu.agent.define({ name: "researcher", system: "AGENT=researcher", memory: "off", rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
  sofuu.agent.define({
    name: "orchestrator",
    system: "AGENT=orch You coordinate.",
    agents: ["researcher"],
    memory: "off", rlm: "off",
    budget: { maxSteps: 4 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("orchestrator", "find the answer", {});
    check("A2 delegation ran a sub-run", r.subRuns.length === 1 && r.subRuns[0].name === "researcher");
    check("A2 child answer reached parent", r.answer.indexOf("orch-final[has-child]") >= 0);
    check("A2 delegate event in trace", r.trace.some(e => e.kind === "delegate" && e.payload && e.payload.agent === "researcher"));
    check("A2 child usage merged into parent", r.usage.llmCalls >= 2);
    check("A2 renderTrace renders both", sofuu.agent.renderTrace(r).indexOf("researcher") >= 0);
  }

  /* A2: depth cap — the child at maxDepth gets NO delegate tool */
  sofuu.agent.define({
    name: "capB",
    system: "AGENT=capB",
    tools: [{ name: "btool", description: "x", parameters: { type: "object", properties: {} }, execute: async () => "bt" }],
    agents: ["capA"], memory: "off", rlm: "off", provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  sofuu.agent.define({
    name: "capA", system: "AGENT=capA You are A.",
    tools: [{ name: "atool", description: "x", parameters: { type: "object", properties: {} }, execute: async () => "at" }],
    agents: ["capB"], memory: "off", rlm: "off",
    budget: { maxSteps: 6, maxDepth: 1 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const events = [];
    const r = await sofuu.agent.run("capA", "go", { onStep: e => events.push(e) });
    const parentPlan = events.find(e => e.kind === "plan" && e.name === "capA");
    const childErr = events.find(e => e.kind === "tool_result" && e.name === "capB" &&
                                      String((e.payload && e.payload.error) || "").indexOf("no such tool: delegate") >= 0);
    check("A2 parent got the delegate tool", !!(parentPlan && parentPlan.payload.tools.indexOf("delegate") >= 0));
    check("A2 child at maxDepth has NO delegate tool (structural)", !!childErr);
    check("A2 depth-cap run completes", r.answer.indexOf("capA-done") >= 0);
  }

  /* A2: cycle guard A→B→A */
  sofuu.agent.define({
    name: "alpha", system: "AGENT=alpha You are alpha.",
    tools: [{ name: "atool2", description: "x", parameters: { type: "object", properties: {} }, execute: async () => "a" }],
    agents: ["beta"], memory: "off", rlm: "off",
    budget: { maxSteps: 6, maxDepth: 3 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  sofuu.agent.define({
    name: "beta", system: "AGENT=beta You are beta.",
    tools: [{ name: "btool2", description: "x", parameters: { type: "object", properties: {} }, execute: async () => "b" }],
    agents: ["alpha"], memory: "off", rlm: "off",
    budget: { maxSteps: 6, maxDepth: 3 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("alpha", "go", {});
    const sawCycle = JSON.stringify(r.trace).indexOf("cycle") >= 0 ||
                     JSON.stringify(r.subRuns.map(s => s.trace)).indexOf("cycle") >= 0;
    check("A2 cycle A→B→A surfaced as tool error", sawCycle);
    check("A2 cycle did not hang (alpha finished)", r.answer.indexOf("alpha-done") >= 0);
  }

  /* ── A5: budgets, cancel, tool timeout ──────────────────────────── */
  sofuu.agent.define({
    name: "loopy", system: "AGENT=loopy",
    tools: [{ name: "get_weather", description: "w", parameters: { type: "object", properties: {} }, execute: async () => "w" }],
    memory: "off", rlm: "off",
    budget: { maxSteps: 3, maxTokens: 1000000, maxWallMs: 60000 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("loopy", "loop forever", {});
    check("A5 budget_steps stop + salvage answer", r.stopped === "budget_steps" && r.steps === 3 &&
          r.answer === "loopy-salvage-summary");
  }
  sofuu.agent.define({
    name: "loopyTk", system: "AGENT=loopy",
    tools: [{ name: "get_weather", description: "w", parameters: { type: "object", properties: {} }, execute: async () => "w" }],
    memory: "off", rlm: "off",
    budget: { maxSteps: 50, maxTokens: 10, maxWallMs: 60000 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("loopyTk", "loop forever", {});
    /* 1 tool round (48 mock tokens > 10 budget) + 1 salvage call. */
    check("A5 budget_tokens stop + salvage answer", r.stopped === "budget_tokens" &&
          r.usage.llmCalls === 2 && r.answer === "loopy-salvage-summary");
  }

  /* Retry-policy classifier — every libcurl transport-error wording must be
   * transient (HTTP/2 stream resets, connection cuts), HTTP 429/5xx and
   * rate-limit phrases too; permanent failures (auth, empty stream, stall)
   * must NOT be retried. */
  {
    const TRANSIENT = [
      "Stream error in the HTTP/2 framing layer",             /* CURLE_HTTP2_STREAM (92), libcurl 8.7.1 */
      "HTTP/2 stream error",                                  /* CURLE_HTTP2_STREAM, older curl */
      "Error in the HTTP2 framing layer",                     /* CURLE_HTTP2 (16), libcurl 8.7.1 */
      "HTTP/2 framing layer error",                           /* CURLE_HTTP2, older wording */
      "Transferred a partial file",                           /* CURLE_PARTIAL_FILE (18), libcurl 8.7.1 */
      "Transfer closed with outstanding read data remaining", /* CURLE_PARTIAL_FILE, older wording */
      "Server returned nothing (no headers, no data)",        /* CURLE_GOT_NOTHING (52), libcurl 8.7.1 */
      "Empty reply from server",                              /* CURLE_GOT_NOTHING, older wording */
      "Failure when receiving data from the peer",            /* CURLE_RECV_ERROR (56) */
      "Failed sending data to the peer",                      /* CURLE_SEND_ERROR (55) */
      "HTTP 429 — rate limited",
      "HTTP 502 — gateway error",
      "Recv failure: Connection reset by peer",
      "Couldn't connect to server",
      "connection timed out",
      "ETIMEDOUT",
      "upstream temporarily overloaded",
    ];
    for (const m of TRANSIENT) check("A5 transient classified: " + m, sofuu.agent.isTransientProviderError(m));
    const PERMANENT = [
      "HTTP 401 — invalid api key",
      "HTTP 404 — model not found",
      "provider returned an empty stream (finish_reason: stop)",
      "provider sent no data for 300s — the model is unresponsive (retry, or switch models)",
      "no such tool: xyz",
    ];
    for (const m of PERMANENT) check("A5 permanent NOT classified: " + m, !sofuu.agent.isTransientProviderError(m));
  }

  /* Retry gate: transient errors are retried ONLY while nothing has been
   * forwarded to the UI yet — a mid-stream failure after content was shown
   * must fail loudly instead of duplicating text by re-streaming. */
  sofuu.agent.define({
    name: "partial429", system: "AGENT=partial429", memory: "off", rlm: "off",
    budget: { maxSteps: 4 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const before = PARTIAL429_HITS;
    let err = null;
    try { await sofuu.agent.run("partial429", "hi", {}); } catch (e) { err = e; }
    check("A5 mid-stream 429 after content: NO retry, cause surfaced",
          !!err && String(err.message).indexOf("HTTP 429") >= 0 && PARTIAL429_HITS === before + 1);
  }
  sofuu.agent.define({
    name: "early429", system: "AGENT=early429", memory: "off", rlm: "off",
    budget: { maxSteps: 4 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const before = EARLY429_HITS;
    let err = null;
    try { await sofuu.agent.run("early429", "hi", {}); } catch (e) { err = e; }
    /* 1 initial attempt + 2 backoff retries (1.5s + 4s) = 3 requests. */
    check("A5 429 before content: retried (3 attempts), cause surfaced",
          !!err && String(err.message).indexOf("HTTP 429") >= 0 && EARLY429_HITS === before + 3);
  }

  sofuu.agent.define({
    name: "tooly", system: "AGENT=tooly",
    tools: [{ name: "slow_tool", description: "hangs", parameters: { type: "object", properties: {} },
              execute: () => new Promise(() => {}) }],
    memory: "off", rlm: "off",
    toolTimeoutMs: 250,
    budget: { maxSteps: 4 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("tooly", "try the hanging tool", {});
    check("A5 tool timeout surfaced, run survived", r.answer.indexOf("tooly-done-saw-timeout") >= 0);
  }

  /* cancel mid-run */
  sofuu.agent.define({ name: "slow", system: "AGENT=slow", memory: "off", rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
  {
    const p = sofuu.agent.run("slow", "slow task A", { signal: "cx1" });
    setTimeout(() => { sofuu.agent.cancel("cx1"); }, 60);
    const r = await p;
    check("A5 cancel stops the run", r.stopped === "cancelled");
  }

  /* ── A2: runMany concurrency ────────────────────────────────────── */
  {
    REQ_LOG.length = 0;
    const rs = await sofuu.agent.runMany(
      [1, 2, 3, 4].map(i => ({ agent: "slow", task: "slow task " + i, opts: PROV })),
      { concurrency: 2 });
    check("A2 runMany all answered", rs.length === 4 && rs.every(r => r.answer === "slow-done"));
    let maxIn = 0;
    for (const a of REQ_LOG) {
      if (!a.end) continue;
      let in_ = 0;
      for (const b of REQ_LOG) {
        if (!b.end) continue;
        if (b.start <= a.start && a.start < b.end) in_++;
      }
      if (in_ > maxIn) maxIn = in_;
    }
    check("A2 concurrency cap respected (max in-flight " + maxIn + " ≤ 2)", maxIn <= 2);
    check("A2 concurrency actually parallel (reached 2)", maxIn === 2);
  }

  /* ── A4: mapContext over a long context ─────────────────────────── */
  sofuu.agent.define({ name: "chunkworker", system: "AGENT=chunkworker", memory: "off", rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
  sofuu.agent.define({ name: "reducer", system: "AGENT=reducer", memory: "off", rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
  {
    const parts = [];
    for (let i = 1; i <= 60; i++) {
      parts.push("Section " + i + "\n" + ("Ordinary prose about topic " + i + ". ").repeat(60));
      if (i === 37) parts.push("IMPORTANT: the map code word is MAP-NEEDLE-8891.");
    }
    const ctx = parts.join("\n\n");
    const r = await sofuu.agent.mapContext(ctx, "find the map code word", {
      agent: "chunkworker", reduceAgent: "reducer", chunkChars: 6000, concurrency: 4,
    });
    check("A4 mapContext merged answer finds needle", r.answer.indexOf("MAP-NEEDLE-8891 present") >= 0);
    check("A4 mapContext subRuns = chunk count (" + r.subRuns.length + ")", r.subRuns.length >= 8 && r.subRuns.length <= 25);
  }

  /* ── A4: RLM fallback + recurseVia ──────────────────────────────── */
  {
    sofuu.agent.define({
      name: "rlmfb", system: "AGENT=simple", rlm: "on", memory: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
      tools: [{ name: "get_weather", description: "Get weather",
                parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
                execute: async (args) => "TOOLRES-9f3 weather for " + (args && args.city) + ": 25C sunny" }],
    });
    const origQuery = sofuu.rlm.query;
    sofuu.rlm.query = () => Promise.reject(new Error("rlm down"));
    const r = await sofuu.agent.run("rlmfb", "weather in Paris?", {});
    sofuu.rlm.query = origQuery;
    check("A4 rlm failure falls back to the plain loop", r.answer.indexOf("weather-final sees TOOLRES-9f3: yes") >= 0);
    check("A4 fallback noted in trace", r.trace.some(e => e.kind === "rlm:fallback"));
  }
  {
    /* rlm 'on' routes the whole turn through sofuu.rlm.query */
    sofuu.agent.define({ name: "rlmy", system: "plain agent", memory: "off", rlm: "on", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    const r = await sofuu.agent.run("rlmy", "hello", {});
    if (r.answer !== "RLM-RECURSE-OK") console.log("  rlm:y answer=" + JSON.stringify(r.answer) + " stopped=" + r.stopped + " trace=" + r.trace.map(e=>e.kind).join(","));
    check("A4 rlm:on turn routed through rlm", r.answer === "RLM-RECURSE-OK");
    check("A4 rlm trace events folded", r.trace.some(e => e.kind.indexOf("rlm:") === 0));
  }
  {
    /* recurseVia: the RLM sandbox's llm() sub-calls run through the
     * tool-capable 'researcher' agent instead of a raw completion. */
    const ctx = "RECURSE-NEEDLE planted at the very start. " + "filler text. ".repeat(2000);
    const r = await sofuu.rlm.query(ctx, "where is the recurse needle?", Object.assign({
      recurseVia: { agent: "researcher" }, maxWallMs: 60000,
    }, PROV));
    if (r.answer !== "RLM-RECURSE-OK") console.log("  recurseVia answer=" + JSON.stringify(r.answer) + " stopped=" + r.stopped + " calls=" + r.calls);
    check("A4 rlm recurseVia ran through the agent", r.answer === "RLM-RECURSE-OK");
  }
  {
    /* A4 FULL FORM: agent tools injected into the RLM sandbox — the
     * model's snippet calls tool(); the host executes it through the
     * agent's normal tool path (timeout/trace/budgets); the snippet
     * resumes with the result. */
    sofuu.agent.define({
      name: "rlmtooly", system: "rlm tool test agent", rlm: "on", memory: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
      tools: [{ name: "get_weather", description: "Get weather",
                parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
                execute: async (args) => "TOOLRES-9f3 weather for " + (args && args.city) + ": 25C sunny" }],
    });
    const r = await sofuu.agent.run("rlmtooly", "use the sandbox tool", {});
    if (r.answer !== "RLM-TOOL-OK") console.log("  rlmtooly answer=" + JSON.stringify(r.answer) + " stopped=" + r.stopped + " trace=" + r.trace.map(e => e.kind).join(","));
    check("A4 full: sandbox tool() executed through the agent path", r.answer === "RLM-TOOL-OK");
    check("A4 full: rlm:tool event folded into the run trace", r.trace.some(e => e.kind === "rlm:tool"));
    check("A4 full: sandbox tool call counted in run usage", r.usage.toolCalls === 1);
    check("A4 full: system prompt documents the tool (mock saw it)", RLM_SYS.indexOf("get_weather") >= 0 && RLM_SYS.indexOf("tool(name, args)") >= 0);
  }

  /* ── web: built-in web_search tool via the endpoint mock ────────── */
  sofuu.agent.define({ name: "webby", system: "AGENT=webby", tools: ["web"], memory: "off", rlm: "off", provider: "openai", model: "mock", api_key: "x", base_url: MOCK });
  {
    const r = await sofuu.agent.run("webby", "search the web", {});
    check("web web_search tool end-to-end", r.answer === "web-final[ok]");
  }

  /* ── F4b: Anthropic-wire stream-first tool calls ────────────────── */
  sofuu.agent.define({
    name: "anth",
    system: "You are the anthropic wire test agent.",
    tools: [{
      name: "get_weather", description: "Get weather",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
      execute: async (args) => "ANTH-TOOLRES-77 weather for " + (args && args.city) + ": 21C",
    }],
    memory: "off", rlm: "off",
    provider: "anthropic", model: "mock", api_key: "x", base_url: ANTH_MOCK,
  });
  {
    const r = await sofuu.agent.run("anth", "weather in Paris?", {});
    check("F4b anthropic stream-first tool call round-trips", r.answer === "anth-final[ok]");
    check("F4b tool_use args merged across input_json_delta fragments",
      r.trace.some(e => e.kind === "tool" && String((e.payload && e.payload.args) || "").indexOf("Paris") >= 0));
    check("F4b anthropic usage mapped (input/output tokens)",
      r.usage.promptTokens > 0 && r.usage.completionTokens > 0 && r.usage.llmCalls === 2);
    /* P6: the anthropic wire carries the stable prefix as a cacheable
     * system block + a marker on the LAST tool; system text intact. */
    check("P6 system rides as a cacheable block", !!(ANT_BODY && Array.isArray(ANT_BODY.system) &&
      ANT_BODY.system.length > 0 &&
      ANT_BODY.system[ANT_BODY.system.length - 1].cache_control &&
      ANT_BODY.system[ANT_BODY.system.length - 1].cache_control.type === "ephemeral"));
    check("P6 last tool carries the cache marker", !!(ANT_BODY && Array.isArray(ANT_BODY.tools) &&
      ANT_BODY.tools.length > 0 &&
      ANT_BODY.tools[ANT_BODY.tools.length - 1].cache_control &&
      ANT_BODY.tools[ANT_BODY.tools.length - 1].cache_control.type === "ephemeral"));
    check("P6 anthropic system text intact after block reshape",
      ANT_SYS.indexOf("anthropic wire test agent") >= 0);
  }
  {
    /* P6: cache usage parses off the SSE usage objects (provider-neutral
     * slots; OpenAI's cached_tokens maps to cacheReadTokens too). */
    const st = sofuu.ai.stream({ provider: "anthropic", model: "mock", api_key: "x",
      base_url: ANTH_MOCK, messages: [{ role: "user", content: "cache probe" }] });
    for await (const c of st) { void c; }
    check("P6 anthropic cache usage parsed (write=333 read=777)",
      st.usage.cacheWriteTokens === 333 && st.usage.cacheReadTokens === 777);
    check("P6 openai wire stays cache_control-free (auto-cache needs nothing)",
      typeof sofuu.ai.stream === "function" && LAST_OPENAI_RAW.indexOf("cache_control") < 0);
  }

  /* ── A6: MCP name→owner routing + inline-wins clash ─────────────── */
  sofuu.agent.define({
    name: "mcpt", system: "AGENT=mcpt",
    tools: [
      { mcp: [
          { name: "srv1", command: RUNTIME + " run examples/mcp_echo_server.js srv1" },
          { name: "srv2", command: RUNTIME + " run examples/mcp_echo_server.js srv2" },
      ] },
      { name: "shared_tool", description: "inline wins", parameters: { type: "object", properties: {} },
        execute: async () => "INLINE-TOOL-WINS" },
    ],
    memory: "off", rlm: "off",
    budget: { maxSteps: 6 }, provider: "openai", model: "mock", api_key: "x", base_url: MOCK
  });
  {
    const r = await sofuu.agent.run("mcpt", "use the tools", {});
    const traceStr = JSON.stringify(r.trace);
    check("A6 srv2_tool routed to srv2 (not first-server srv1)", traceStr.indexOf("echo from srv2/srv2_tool") >= 0);
    check("A6 inline tool wins the name clash", r.answer.indexOf("mcp-final[inline-ok]") >= 0);
    check("A6 clash warning in trace", r.trace.some(e => e.kind === "warn" && JSON.stringify(e.payload).indexOf("clash") >= 0));
  }

  /* ── A3: memory scopes on one brain file (QTSQ builds only) ─────── */
  let brainOk = false;
  const TMP_BRAIN = "/tmp/sofuu_agent_test_brain_" + Date.now() + ".qtsq";
  try {
    if (sofuu.memory && sofuu.memory.open && sofuu.ai && sofuu.ai.embedLocal) {
      const dim = sofuu.ai.embedLocal("").length;
      const b = sofuu.memory.open(TMP_BRAIN, dim);
      b.remember(new Float32Array(dim).map((_, i) => (i % 7) / 7), "probe text", "user", 0);
      b.flush();
      brainOk = typeof b.count === "function";
    }
  } catch (e) { brainOk = false; }
  if (!brainOk) {
    skip("A3 memory scopes", "sofuu.memory unavailable (no-QTSQ build)");
  } else {
    sofuu.agent.define({ name: "memA", system: "AGENT=memA", memory: "agent", brainPath: TMP_BRAIN, rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    sofuu.agent.define({ name: "memB", system: "AGENT=memB", memory: "agent", brainPath: TMP_BRAIN, rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    sofuu.agent.define({ name: "memOff", system: "AGENT=memOff", memory: "off", brainPath: TMP_BRAIN, rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });

    await sofuu.agent.run("memA", "remember the secret code MEM-XYZ-123", {});
    {
      const evB = [];
      await sofuu.agent.run("memB", "what is the secret code?", { onStep: e => evB.push(e) });
      const recallB = evB.find(e => e.kind === "recall");
      check("A3 agent scope does not leak across agents", !recallB || (recallB.payload.count || 0) === 0);
    }
    {
      const evA = [];
      await sofuu.agent.run("memA", "what is the secret code?", { onStep: e => evA.push(e) });
      const recallA = evA.find(e => e.kind === "recall");
      check("A3 own scope recalls stored memories", !!recallA && recallA.payload.count >= 1);
      /* P2: gating is observable — dropped/budgetCut counts + hit list. */
      check("P2 recall event reports gating counts", !!recallA &&
        typeof recallA.payload.dropped === "number" &&
        typeof recallA.payload.budgetCut === "number" &&
        Array.isArray(recallA.payload.hits) &&
        recallA.payload.hits.every(h => typeof h.score === "number"));
    }
    {
      const b = sofuu.memory.open(TMP_BRAIN, sofuu.ai.embedLocal("").length);
      const before = b.count();
      await sofuu.agent.run("memOff", "must not write anything", {});
      const after = b.count();
      check("A3 memory:off writes nothing (" + before + " → " + after + ")", after === before);
    }

    /* ── Consolidation cadence (now wired into brainDecayTick): a brain
     *    with ≥24 weak-old episodic records consolidates at the turn
     *    boundary (event + persisted semantic records + shrunken count),
     *    and is rate-limited — a second run inside the interval does
     *    nothing. Eligibility gate: episodic, strength < 0.25, age > 1d. */
    {
      const TMP_CON = "/tmp/sofuu_agent_test_con_" + Date.now() + ".qtsq";
      const dim = sofuu.ai.embedLocal("").length;
      const cb = sofuu.memory.open(TMP_CON, dim);
      for (let i = 0; i < 30; i++) {
        const t = "consol episode " + i + " about the project alpha migration " + (i % 2 ? "database" : "api");
        cb.remember(new Float32Array(sofuu.ai.embedLocal(t)), t, "user", 0);
      }
      cb.decayTick(Math.floor(1.5 * 86400)); /* strength ≈ 0.22 — weak but above the retain floor */
      cb.flush();
      const before = cb.count();
      sofuu.agent.define({ name: "conso", system: "AGENT=mem", memory: "shared", brainPath: TMP_CON,
        rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
      const ev = [];
      await sofuu.agent.run("conso", "tell me about project alpha", { onStep: e => ev.push(e) });
      const conEv = ev.find(e => e.kind === "consolidated");
      check("consolidation fires at the turn boundary (event, ≥1 cluster)", !!conEv && (conEv.payload.clusters || 0) >= 1);
      const rb = sofuu.memory.open(TMP_CON, dim);
      const tiers = {};
      rb.recall(new Float32Array(sofuu.ai.embedLocal("project alpha migration")), 32)
        .forEach(h => { tiers[h.tier] = (tiers[h.tier] || 0) + 1; });
      check("semantic (tier 2) records exist and persisted", (tiers[2] || 0) >= 1);
      const ev2 = [];
      await sofuu.agent.run("conso", "one more question for the interval probe", { onStep: e => ev2.push(e) });
      check("consolidation rate-limited inside the interval", !ev2.some(e => e.kind === "consolidated"));
      /* Consolidation tombstones the originals; the NEXT turn boundary's
       * retain() physically prunes them (decay→retain→consolidate order),
       * so the shrunken count is observable after the second run. */
      const rb2 = sofuu.memory.open(TMP_CON, dim);
      check("merged originals pruned (count " + before + " → " + rb2.count() + ")", rb2.count() < before);
      try { sofuu.fs.rm(TMP_CON); } catch (e) {}
    }
  }

  /* ── Built-in coding tools (src/js/tools.js) ──────────────────── */
  {
    check("code tools registered (7 builtins)",
      sofuu.tools && sofuu.tools.TOOLS &&
      ["read_file", "write_file", "edit_file", "grep", "glob", "list_dir", "bash"]
        .every(n => sofuu.tools.TOOLS[n]));
    const CTD = "/tmp/sofuu_codetools_" + Date.now();
    await sofuu.fs.mkdir(CTD, { recursive: true });
    const prevCwd = process.cwd();
    process.chdir(CTD);
    try {
      sofuu.agent.define({ name: "coder", system: "AGENT=codetool", tools: ["code"],
        rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model,
        api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
        budget: { maxSteps: 8 } });
      const r = await sofuu.agent.run("coder", "build the thing", {});
      check("coder ran all 5 tool rounds",
        r.trace.filter(e => e.kind === "tool").length === 5);
      check("coder E2E write/read/edit/grep/bash verified via results",
        r.answer === "codetool-final[+write+read+edit+grep+bash]");

      sofuu.agent.define({ name: "coderail", system: "AGENT=coderail", tools: ["code"],
        rlm: "off", provider: DEF_PROV.provider, model: DEF_PROV.model,
        api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
        budget: { maxSteps: 8 } });
      const rr = await sofuu.agent.run("coderail", "try the rails", {});
      check("rails: ambiguous edit refuses, not-found errors, jail holds, bash timeout kills",
        rr.answer === "coderail-final[+ambiguous+notfound+jailed+timeout]");
    } finally { process.chdir(prevCwd); }
  }

  /* ── A8: loadDir + list ─────────────────────────────────────────── */
  {
    const dir = "/tmp/sofuu_agent_defs_" + Date.now();
    await sofuu.fs.mkdir(dir);
    await sofuu.fs.writeFile(dir + "/good.js",
      'sofuu.agent.define({ name: "loadme", system: "loaded from disk", memory: "off" });');
    await sofuu.fs.writeFile(dir + "/broken.js", 'throw new Error("bad agent file");');
    const ld = await sofuu.agent.loadDir(dir);
    check("A8 loadDir loads the good file", ld.loaded.indexOf("loadme") >= 0);
    check("A8 loadDir lists the broken file, never fatal",
      ld.broken.length === 1 && ld.broken[0].file === "broken.js" && ld.broken[0].error.indexOf("bad agent file") >= 0);
    const names = sofuu.agent.list().map(a => a.name);
    check("A8 list() shows registry", names.indexOf("simple") >= 0 && names.indexOf("loadme") >= 0);
    /* re-load is safe: duplicate define lands in broken, not a crash */
    const ld2 = await sofuu.agent.loadDir(dir);
    check("A8 duplicate define is reported, not fatal", ld2.broken.length >= 1);
  }

  /* ── Per-model context windows (RLM routing) ────────────────────── */
  {
    /* Table lookups via the read-only helper. */
    check("CTX claude-sonnet-4-5 → 200000", sofuu.agent.contextWindow("claude-sonnet-4-5") === 200000);
    check("CTX gpt-4o → 128000", sofuu.agent.contextWindow("gpt-4o") === 128000);
    check("CTX gpt-4.1 → 1000000", sofuu.agent.contextWindow("gpt-4.1") === 1000000);
    check("CTX grok-4 → 256000", sofuu.agent.contextWindow("grok-4") === 256000);
    check("CTX llama3 → 8192", sofuu.agent.contextWindow("llama3") === 8192);
    check("CTX openrouter-prefixed model matches ('openai/gpt-4o')",
      sofuu.agent.contextWindow("openai/gpt-4o") === 128000);
    check("CTX unknown model → 32768 default", sofuu.agent.contextWindow("totally-unknown-99") === 32768);
    check("CTX empty model → 32768 default", sofuu.agent.contextWindow("") === 32768);

    /* Routing integration: a small-window model (llama3 → 8192) must route
     * an oversized turn through RLM where the old flat 32768 default would
     * have routed it plain. Non-holistic question so ONLY the >80% rule
     * can fire (discriminative against the old default). */
    const filler = ("filler line " + "x".repeat(64) + "\n").repeat(780); // ≈57KB ≈ 14k tokens
    const evs = [];
    sofuu.agent.define({
      name: "ctxwin", system: "AGENT=ctxwin", memory: "off",
      provider: DEF_PROV.provider, model: "llama3", api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
      rlm: "auto",
    });
    await sofuu.agent.run("ctxwin", "what is the secret code?\n\n" + filler,
      { onStep: e => evs.push(e) });
    const rr = evs.find(e => e.kind === "rlm:route");
    check("CTX oversized turn routes rlm on llama3's 8192 window", !!rr && rr.payload.route === "rlm");
    check("CTX routed windowTokens === 8192 (per-model, not flat 32768)",
      !!rr && rr.payload.windowTokens === 8192);
  }

  /* ═══ PLAN-MEMORY-TOKENS: P1 core prompt · P3 truncation · P4 single
   * task-embed · M4 trace cap ═══════════════════════════════════════ */
  check("P1 CORE_PROMPT exported", typeof sofuu.agent.CORE_PROMPT === "string" &&
    sofuu.agent.CORE_PROMPT.indexOf("Sofuu coding agent") === 0);
  {
    /* P1: a def with NO custom system sends exactly the core prompt —
     * no persona padding, no name boilerplate. */
    sofuu.agent.define({ name: "p1bare", memory: "off", rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    let seen = null;
    const evs1 = [];
    await sofuu.agent.run("p1bare", "hello", { onStep: e => { evs1.push(e); } });
    /* the plain path streams; probe via a second run that captures sys in
     * decide() through RLM_SYS (system-first calls land there only for
     * rlm) — simplest reliable capture: run once more and read the mock's
     * LAST_OPENAI_RAW. */
    await sofuu.agent.run("p1bare", "capture system line", {});
    try { seen = JSON.parse(LAST_OPENAI_RAW).messages.find(m => m.role === "system").content; } catch (e) {}
    check("P1 default system is exactly CORE_PROMPT (+identity only)", !!seen &&
      seen.indexOf("Sofuu coding agent") === 0 && seen.indexOf("helpful agent named") < 0);
    /* P1: a custom def.system COMPOSES AFTER the core — the discipline
     * lines survive user overrides (the plan's small-model escape hatch). */
    sofuu.agent.define({ name: "p1custom", system: "AGENT=p1custom Extra scaffolding.", memory: "off", rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    await sofuu.agent.run("p1custom", "capture composed system", {});
    let seen2 = null;
    try { seen2 = JSON.parse(LAST_OPENAI_RAW).messages.find(m => m.role === "system").content; } catch (e) {}
    check("P1 custom system composes AFTER the core", !!seen2 &&
      seen2.indexOf("Sofuu coding agent") === 0 &&
      seen2.indexOf("AGENT=p1custom Extra scaffolding.") > 0);
    /* The chat def passes CORE_PROMPT itself — must not duplicate. */
    sofuu.agent.define({ name: "p1chat", system: sofuu.agent.CORE_PROMPT, memory: "off", rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    await sofuu.agent.run("p1chat", "capture chat-shaped system", {});
    let seen3 = null;
    try { seen3 = JSON.parse(LAST_OPENAI_RAW).messages.find(m => m.role === "system").content; } catch (e) {}
    check("P1 passing CORE_PROMPT as system is a no-op (no duplication)", !!seen3 &&
      seen3 === sofuu.agent.CORE_PROMPT);
  }
  {
    /* P3: a 48k-char tool result enters context capped at ~4k with head +
     * tail + recovery marker; trace records honest chars/kept counts. */
    sofuu.agent.define({
      name: "bigtool", system: "AGENT=bigtool", memory: "off", rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
      tools: [{ name: "big_tool", description: "returns a huge payload",
                parameters: { type: "object", properties: {} },
                execute: async () => "HEADMARKER-" + "x".repeat(48000) + "-TAILMARKER" }],
    });
    const r = await sofuu.agent.run("bigtool", "fetch the big payload", {});
    check("P3 huge tool result truncated (head+tail+marker kept)",
      r.answer.indexOf("capped") >= 0 && r.answer.indexOf("+head") >= 0 && r.answer.indexOf("+tail") >= 0);
    const tr = r.trace.find(e => e.kind === "tool_result" && e.payload && e.payload.chars > 40000);
    check("P3 trace records chars/kept (" + (tr && tr.payload.chars) + "→" + (tr && tr.payload.kept) + ")",
      !!tr && tr.payload.kept <= 4000 && tr.payload.kept > 3000);
  }
  {
    /* P4: one embed per turn for the task (recall-time vec reused at
     * store), not two. */
    const brain4 = "/tmp/sofuu_p4_brain_" + Date.now() + ".qtsq";
    sofuu.agent.define({ name: "p4probe", system: "AGENT=mem", memory: "agent",
      brainPath: brain4, rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    let taskEmbeds = 0;
    const origEmbed = sofuu.ai.embedLocal;
    sofuu.ai.embedLocal = function (t) {
      if (String(t).indexOf("P4EMBEDPROBE-7788") >= 0) taskEmbeds++;
      return origEmbed.call(sofuu.ai, t);
    };
    try {
      await sofuu.agent.run("p4probe", "P4EMBEDPROBE-7788 unique task sentence about quantum flux patterns", {});
    } finally {
      sofuu.ai.embedLocal = origEmbed;
    }
    check("P4 task embedded exactly once per turn (was 2)", taskEmbeds === 1);
  }
  {
    /* M4: the trace keeps the FULL timeline as {kind,t} stubs but bounds
     * the heavy payloads to TRACE_CAP (512) — first-16 skeleton + freshest
     * 496 keep payloads, the evicted middle is demoted. A long tool loop
     * can't balloon retained payload data. */
    sofuu.agent.define({
      name: "ringcap", system: "AGENT=loopy", memory: "off", rlm: "off",
      tools: [{ name: "get_weather", description: "w", parameters: { type: "object", properties: {} }, execute: async () => "w" }],
      budget: { maxSteps: 400, maxTokens: 100000000, maxWallMs: 120000 },
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const r = await sofuu.agent.run("ringcap", "loop hard", {});
    const withPayload = r.trace.filter(e => e.payload !== undefined).length;
    const stubs = r.trace.filter(e => e.payload === undefined).length;
    check("M4 timeline preserved past the cap (got " + r.trace.length + " events)",
      r.stopped === "budget_steps" && r.trace.length > 512);
    check("M4 heavy payloads bounded to 512 (got " + withPayload + " with payload, " + stubs + " stubs)",
      withPayload === 512 && stubs > 0);
    check("M4 skeleton preserved (start first, keeps payload)",
      r.trace[0].kind === "start" && r.trace[0].payload !== undefined);
    check("M4 freshest event keeps its payload",
      r.trace[r.trace.length - 1].payload !== undefined);
  }

  /* ═══ PLAN-ML-GATES phase 1: date fix + sofuu.ml surface + the
   * rule-based supervisor subset (dup calls, re-reads of unchanged
   * files) ═══════════════════════════════════════════════════════════ */
  {
    /* §7: today's date enters the wire request in the EPHEMERAL user-role
     * context message — never in the byte-stable system prompt. */
    sofuu.agent.define({ name: "mldate", memory: "off", rlm: "off",
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url });
    await sofuu.agent.run("mldate", "what day is it", {});
    let dateMsgs = null;
    try { dateMsgs = JSON.parse(LAST_OPENAI_RAW).messages; } catch (e) {}
    const dateUser = !!dateMsgs && dateMsgs.find(m => m.role === "user" &&
      /Current date: \d{4}-\d{2}-\d{2}\./.test(String(m.content || "")));
    const dateSys = !!dateMsgs && String((dateMsgs.find(m => m.role === "system") || {}).content || "");
    check("ML date fix: 'Current date: YYYY-MM-DD' rides a user message", !!dateUser);
    check("ML date fix: system prompt stays date-free (byte-stable, P1/P6)",
      !!dateMsgs && dateSys.indexOf("Current date") < 0);
  }
  if (sofuu.ml && typeof sofuu.ml.track === "function") {
    check("ML surface: sofuu.ml.{track,workset,info,feedback,supervisor.check,supervisor.loop} present",
      typeof sofuu.ml.workset === "function" && typeof sofuu.ml.info === "function" &&
      typeof sofuu.ml.feedback === "function" &&
      !!sofuu.ml.supervisor && typeof sofuu.ml.supervisor.check === "function" &&
      typeof sofuu.ml.supervisor.loop === "function");
    let info = null;
    try { info = JSON.parse(sofuu.ml.info()); } catch (e) {}
    check("ML info(): version + gate states reported (supervisor trained = 'on')",
      !!info && info.version === 1 && info.gates && info.gates.supervisor === "on");
    check("ML info(): online-learning block reported (off by default, §13)",
      !!info && !!info.online && info.online.enabled === false &&
      typeof info.online.examples === "number");

    /* §5: the trained freshness gate — direct API shape + verdicts on
     * hand-held material (same cases as the committed-weights fixtures). */
    check("ML surface: sofuu.ml.freshness.score present",
      !!sofuu.ml.freshness && typeof sofuu.ml.freshness.score === "function");
    check("ML info(): freshness gate reports 'on'",
      !!info && info.gates.freshness === "on");
    const STALE_DOC = "Zephyr-db administration guide. Updated on March 4, 2019. " +
      "This package is deprecated and no longer maintained; the repository was " +
      "archived after support ended in 2020.";
    let fStale = null, fTimeless = null, fTL = null;
    try {
      fStale = JSON.parse(sofuu.ml.freshness.score(STALE_DOC,
        "what is the current recommended way to administer a Zephyr-db cluster?",
        JSON.stringify({ kind: "web" })));
      fTimeless = JSON.parse(sofuu.ml.freshness.score(
        "A B-tree is a fundamental data structure used in systems programming. " +
        "This article explains the invariants, the core operations, and the " +
        "classic trade-offs of the B-tree. Complexity analysis included.",
        "what is the latest best practice for B-tree page splitting?",
        JSON.stringify({ kind: "web" })));
      fTL = JSON.parse(sofuu.ml.freshness.score(STALE_DOC,
        "summarize the writing style of this documentation page",
        JSON.stringify({ kind: "web" })));
    } catch (e) {}
    check("ML freshness: old deprecated doc + time-sensitive task → stale with evidence",
      !!fStale && fStale.stale === true && typeof fStale.score === "number" &&
      String(fStale.reason || "").length > 0);
    check("ML freshness: timeless content stays quiet",
      !!fTimeless && fTimeless.stale === false);
    check("ML freshness: stale doc + timeless task stays quiet (the interaction)",
      !!fTL && fTL.stale === false);

    /* §6: the trained relevance gate — direct API shape + a realistic
     * pre-retrieval menu (same construction as the committed-weights
     * fixtures; payments domain is absent from every training family).
     * Advise-only: plan() returns use/skip ids + scores, never mutates the
     * menu. Indices: 0 definition site (use); 1 off-topic note; 2 license;
     * 3 call-site mention; 4 near-dup of the kept 0; 5 one-word trap — all
     * of 1..5 must be advised skip. */
    check("ML surface: sofuu.ml.relevance.plan present",
      !!sofuu.ml.relevance && typeof sofuu.ml.relevance.plan === "function");
    check("ML info(): relevance gate reports 'on'",
      !!info && info.gates.relevance === "on");
    const REL_TASK = "fix the verify_signature function in the payment webhook handler";
    const REL_DEF = "fn verify_signature checks the payment webhook payload against " +
      "the shared secret and rejects the delivery when the digest does not match";
    const REL_CANDS = [
      { text: REL_DEF, kind: "file", strength: 0, role: 0, path: "src/payments/webhook.rs" },
      { text: "meeting notes from the design review: the palette choices stay as discussed, follow up next week", kind: "other", strength: 0, role: 0, path: "" },
      { text: "permission is hereby granted, free of charge, to any person obtaining a copy of this software. the software is provided as is, without warranty of any kind. all rights reserved.", kind: "file", strength: 0, role: 0, path: "" },
      { text: "the billing exporter calls verify_signature once at startup and otherwise only formats the csv rows", kind: "file", strength: 0, role: 0, path: "src/billing/export.rs" },
      { text: REL_DEF, kind: "file", strength: 0, role: 0, path: "src/payments/webhook.rs" },
      { text: "facilities notice: the payment for the new office plants is due; the delivery of the ferns is scheduled for monday", kind: "other", strength: 0, role: 0, path: "" },
    ];
    let relPlan = null;
    try {
      relPlan = JSON.parse(sofuu.ml.relevance.plan(
        JSON.stringify({ task: REL_TASK, recent: "", candidates: REL_CANDS }),
        JSON.stringify({ kept: [0] })));
    } catch (e) {}
    check("ML relevance: definition site use; off-topic/license/mention/dup/trap skip",
      !!relPlan && Array.isArray(relPlan.use) && Array.isArray(relPlan.skip) &&
      relPlan.use.indexOf(0) >= 0 &&
      [1, 2, 3, 4, 5].every(i => relPlan.skip.indexOf(i) >= 0));
    check("ML relevance: use advice best-first + one score per candidate",
      !!relPlan && relPlan.use[0] === 0 &&
      Array.isArray(relPlan.scores) && relPlan.scores.length === REL_CANDS.length);
    let relPlan2 = null;
    try {
      relPlan2 = JSON.parse(sofuu.ml.relevance.plan(
        JSON.stringify({ task: REL_TASK, recent: "", candidates: REL_CANDS }),
        JSON.stringify({ kept: [0] })));
    } catch (e) {}
    check("ML relevance: plan is deterministic for a fixed menu",
      !!relPlan && !!relPlan2 &&
      JSON.stringify(relPlan.scores) === JSON.stringify(relPlan2.scores) &&
      JSON.stringify(relPlan.use) === JSON.stringify(relPlan2.use) &&
      JSON.stringify(relPlan.skip) === JSON.stringify(relPlan2.skip));

    /* §5/§16 E2E: stale tool material → ONE notice on the next context
     * boundary (the mock model reports seeing it); fresh material → silence. */
    const freshTool = [{ name: "fetch_docs", description: "fetch docs",
      parameters: { type: "object", properties: { topic: { type: "string" } } },
      execute: async (a) => a.topic === "zephyr"
        ? STALE_DOC
        : "Just shipped: the latest release rolled out this week and is now " +
          "available. Announced today; actively developed with weekly releases. " +
          "Updated March 4, 2026." }];
    sofuu.agent.define({
      name: "mlfresh", system: "AGENT=mlfresh", memory: "off", rlm: "off", tools: freshTool,
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const rf = await sofuu.agent.run("mlfresh",
      "what is the current recommended way to administer a Zephyr-db cluster?", {});
    check("ML freshness E2E: notice reached the model (answer saw it)",
      rf.answer.indexOf("+fresh") >= 0);
    const gateF = rf.trace.find(e => e.kind === "mlgate" && e.payload.rule === "freshness");
    check("ML freshness E2E: mlgate event labelled freshness", !!gateF);

    sofuu.agent.define({
      name: "mlfreshclean", system: "AGENT=mlfreshclean", memory: "off", rlm: "off", tools: freshTool,
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const rc = await sofuu.agent.run("mlfreshclean",
      "what is the current state of the Rivermesh gateway?", {});
    check("ML freshness E2E: fresh material stays quiet",
      rc.answer.indexOf("-clean") >= 0 &&
      !rc.trace.some(e => e.kind === "mlgate" && e.payload.rule === "freshness"));

    /* §6/§16 E2E: the pre-retrieval relevance advisor over recalled memory.
     * A brain seeded with on-task records + high-lexical-overlap wrong-topic
     * traps recalls a mixed menu; the advisor must render ONE [relevance]
     * guidance notice on the ephemeral context message (the mock reports
     * seeing it) + an mlgate event. Advise-only — the recall block itself is
     * untouched. Needs QTSQ (sofuu.memory); auto-skips otherwise. */
    if (sofuu.memory && sofuu.memory.open && sofuu.ai && sofuu.ai.embedLocal) {
      const REL_BRAIN = "/tmp/sofuu_agent_test_rel_" + Date.now() + ".qtsq";
      const relDim = sofuu.ai.embedLocal("").length;
      const rb2 = sofuu.memory.open(REL_BRAIN, relDim);
      const relSeed = [
        /* on-task: dense overlap with the run task below */
        "the database migration script rewrites the storage schema and rebuilds every index before the database is swapped live",
        "to fix the migration, check the schema rewrite step in the database script that applies each table change",
        /* wrong-topic traps: share the task's keywords in a different sense */
        "the database migration lunch was catered by the new vendor and the menu plus delivery time for the team were confirmed",
        "the migration of the office plants to the new floor was scheduled by facilities and the database of lunch orders was updated",
      ];
      for (const t of relSeed) rb2.remember(new Float32Array(sofuu.ai.embedLocal(t)), t, "user", 0);
      rb2.flush();
      sofuu.agent.define({
        name: "mlrel", system: "AGENT=mlrel", memory: "shared", brainPath: REL_BRAIN, rlm: "off",
        provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
      });
      const rrel = await sofuu.agent.run("mlrel",
        "fix the database migration script that rewrites the schema", {});
      check("ML relevance E2E: guidance notice reached the model (answer saw it)",
        rrel.answer.indexOf("+rel") >= 0);
      const gateRel = rrel.trace.find(e => e.kind === "mlgate" && e.payload.rule === "relevance");
      check("ML relevance E2E: mlgate event labelled relevance with use/skip counts",
        !!gateRel && typeof gateRel.payload.use === "number" && typeof gateRel.payload.skip === "number" &&
        gateRel.payload.skip >= 1);
    } else {
      skip("ML relevance E2E", "sofuu.memory unavailable (no-QTSQ build)");
    }

    /* §11 rule subset — exact repeat call: the second identical call gets
     * an in-band nudge on its tool result + an mlgate trace event. The
     * call still RUNS (advisors never block). */
    sofuu.agent.define({
      name: "mlrep", system: "AGENT=mlrep", memory: "off", rlm: "off",
      tools: [{ name: "get_weather", description: "w",
                parameters: { type: "object", properties: { city: { type: "string" } } },
                execute: async (a) => "WEATHER-" + a.city }],
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const r1 = await sofuu.agent.run("mlrep", "weather check", {});
    check("ML dup-call: nudge reached the model (answer saw it)",
      r1.answer.indexOf("+nudge") >= 0);
    const gate1 = r1.trace.find(e => e.kind === "mlgate");
    check("ML dup-call: mlgate event emitted with rule label",
      !!gate1 && gate1.payload.rule === "dup_call" && gate1.payload.step === 2);
    const tr1 = r1.trace.filter(e => e.kind === "tool_result");
    check("ML dup-call: first result clean, second carries [supervisor:]",
      tr1.length === 2 &&
      String(tr1[0].payload.result).indexOf("[supervisor:") < 0 &&
      String(tr1[1].payload.result).indexOf("[supervisor:") >= 0);

    /* §11 rule subset — re-reading an unchanged file (different args so
     * the dup rule stays quiet). */
    sofuu.agent.define({
      name: "mlreread", system: "AGENT=mlreread", memory: "off", rlm: "off",
      tools: [{ name: "read_file", description: "r",
                parameters: { type: "object", properties: { path: { type: "string" }, offset: { type: "number" } } },
                execute: async (a) => "FILE-CONTENT " + a.path }],
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const r2 = await sofuu.agent.run("mlreread", "read the file", {});
    check("ML reread: nudge reached the model (answer saw it)",
      r2.answer.indexOf("+reread") >= 0);
    const gate2 = r2.trace.find(e => e.kind === "mlgate");
    check("ML reread: mlgate event labelled reread_unchanged",
      !!gate2 && gate2.payload.rule === "reread_unchanged");

    /* ml:'off' disables every gate for the def — same scripted repeat,
     * no nudge, no mlgate events. */
    sofuu.agent.define({
      name: "mloff", system: "AGENT=mlrep", memory: "off", rlm: "off", ml: "off",
      tools: [{ name: "get_weather", description: "w",
                parameters: { type: "object", properties: { city: { type: "string" } } },
                execute: async (a) => "WEATHER-" + a.city }],
      provider: DEF_PROV.provider, model: DEF_PROV.model, api_key: DEF_PROV.api_key, base_url: DEF_PROV.base_url,
    });
    const r3 = await sofuu.agent.run("mloff", "weather check again", {});
    check("ML off: no nudge injected when ml:'off'",
      r3.answer.indexOf("-nonudge") >= 0 && !r3.trace.some(e => e.kind === "mlgate"));

    /* ═══ PLAN-ML-GATES phase 5: the trained supervisor (§11) + the
     * online-learning surface (§13). Direct API checks here — seeded
     * through the same check→track flow the agent loop uses. ═══════ */
    const SUP_TASK = "calibrate the weather station sensors before the storm season";
    const SUP_P0 = "src/weather/calibrate.rs";
    const SUP_P1 = "src/weather/sensors.rs";
    function supCheck(run, step, tool, sig, target, extra) {
      return JSON.parse(sofuu.ml.supervisor.check(JSON.stringify(Object.assign({
        run: run, step: step, tool: tool, sig: sig, target: target,
        argsText: "", skipTargets: [], budget: 20, task: SUP_TASK,
      }, extra || {}))));
    }
    /* Seed a run the way agent.js does: run_start, then per call
     * supervisor.check BEFORE and ml.track(tool_result) AFTER. */
    function supSeed(run, calls) {
      sofuu.ml.track(JSON.stringify({ kind: "run_start", run: run, task: SUP_TASK }));
      const verdicts = [];
      for (let i = 0; i < calls.length; i++) {
        const c = calls[i];
        verdicts.push(supCheck(run, i + 1, c.tool, c.sig, c.target, c.extra));
        sofuu.ml.track(JSON.stringify({ kind: "tool_result", run: run, step: i + 1,
          tool: c.tool, target: c.target, chars: c.chars || 500, error: false }));
      }
      return verdicts;
    }
    const readSig = (p) => 'read_file:{"path":"' + p + '"}';

    /* §11 layering: the rule layer speaks first — an exact repeat is
     * flagged with source "rule", and the verdict carries the full
     * shape (ok/reason/nudge/score/source). */
    supSeed("e2e-sup-rule", [{ tool: "read_file", sig: readSig(SUP_P0), target: SUP_P0, chars: 1200 }]);
    const vDup = supCheck("e2e-sup-rule", 2, "read_file", readSig(SUP_P0), SUP_P0);
    check("supervisor check: exact repeat flagged by the RULE layer (dup_call)",
      vDup.ok === false && vDup.reason === "dup_call" && vDup.source === "rule" &&
      typeof vDup.nudge === "string" && vDup.nudge.length > 0 &&
      typeof vDup.score === "number" && vDup.score >= 0 && vDup.score <= 1);

    /* §11: where the rules are silent, the NET speaks — a first call
     * that echoes nothing of the task is off-task waste. */
    sofuu.ml.track(JSON.stringify({ kind: "run_start", run: "e2e-sup-model", task: SUP_TASK }));
    const vOff = supCheck("e2e-sup-model", 1, "read_file",
      readSig("docs/office_snack_poll.md"), "docs/office_snack_poll.md");
    check("supervisor check: zero-echo first call flagged by the MODEL layer",
      vOff.ok === false && vOff.source === "model" && vOff.reason === "off_task" &&
      vOff.score >= 0.82);

    /* §11: the clean side — the task's core file as the first call
     * passes silently (advisors must not nag at real work). */
    sofuu.ml.track(JSON.stringify({ kind: "run_start", run: "e2e-sup-clean", task: SUP_TASK }));
    const vClean = supCheck("e2e-sup-clean", 1, "read_file", readSig(SUP_P0), SUP_P0);
    check("supervisor check: on-task first call passes (ok, no nudge)",
      vClean.ok === true && vClean.nudge === null);

    /* §11 loop boundary: a run hammering one file with offset reads and
     * no writes gets flagged at the boundary; a read→grep→edit run with
     * progress stays silent. */
    const spins = [];
    for (let k = 0; k < 4; k++) {
      spins.push({ tool: "read_file", target: SUP_P1, chars: 40,
        sig: 'read_file:{"offset":' + (k * 50) + ',"path":"' + SUP_P1 + '"}' });
    }
    supSeed("e2e-sup-loop", spins);
    const vLoop = JSON.parse(sofuu.ml.supervisor.loop(
      JSON.stringify({ run: "e2e-sup-loop", step: 5, budget: 20 })));
    check("supervisor loop: spinning run flagged at the boundary (loop_*)",
      vLoop.ok === false && vLoop.source === "model" &&
      String(vLoop.reason).indexOf("loop_") === 0);
    supSeed("e2e-sup-loop-ok", [
      { tool: "read_file", sig: readSig(SUP_P0), target: SUP_P0, chars: 1200 },
      { tool: "grep", sig: 'grep:{"pattern":"sensor_offset"}', target: "sensor_offset", chars: 500 },
      { tool: "edit_file", sig: 'edit_file:{"path":"' + SUP_P0 + '"}', target: SUP_P0, chars: 60 },
    ]);
    const vLoopOk = JSON.parse(sofuu.ml.supervisor.loop(
      JSON.stringify({ run: "e2e-sup-loop-ok", step: 4, budget: 20 })));
    check("supervisor loop: healthy read→grep→edit run stays silent",
      vLoopOk.ok === true);

    /* §13 surface: outcome labels land via sofuu.ml.feedback; unknown
     * (run,step) pairs are refused, and info() counts the example. */
    const fb1 = JSON.parse(sofuu.ml.feedback(JSON.stringify(
      { kind: "outcome", model: "supervisor", run: "e2e-sup-rule", step: 2, wasted: true })));
    check("ml.feedback: outcome label for an observed call accepted", fb1.ok === true);
    const fb2 = JSON.parse(sofuu.ml.feedback(JSON.stringify(
      { kind: "outcome", model: "supervisor", run: "no-such-run", step: 99, wasted: false })));
    check("ml.feedback: label for an unknown call refused", fb2.ok === false);
    let info2 = null;
    try { info2 = JSON.parse(sofuu.ml.info()); } catch (e) {}
    check("ml.info(): labeled example counted (online learning stays off until /ml)",
      !!info2 && !!info2.online && info2.online.enabled === false &&
      info2.online.examples >= 1 && info2.online.observations >= 1);

    /* §8: the working set accumulated the runs above. */
    let ws = null;
    try { ws = JSON.parse(sofuu.ml.workset()); } catch (e) {}
    check("ML workset: runs and calls accumulated (" +
      (ws ? ws.runs + " runs, " + ws.calls + " calls" : "unparsable") + ")",
      !!ws && ws.runs >= 3 && ws.calls >= 6);
  } else {
    skip("ML gates", "sofuu.ml not registered (SOFUU_NO_ML set?)");
  }

  console.log(failures === 0
    ? "\nAGENT TEST: ALL PASSED" + (skips ? " (" + skips + " skipped)" : "")
    : "\nAGENT TEST: " + failures + " FAILURE(S)");
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => {
  console.error("FAIL exception: " + (e && e.stack || e));
  process.exit(1);
});
