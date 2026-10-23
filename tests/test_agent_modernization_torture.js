// tests/test_agent_modernization_torture.js -- Phase 6 integration torture.
//
// 50 agent runs, all in flight at once, each exercising every modernization
// track in one trajectory:
//   track 1  in-process sandbox      code_eval computes a run-specific token
//   track 2  native context          an inline tool reads sofuu.agent.current()
//   track 3  cooperative scheduling  50 concurrent streams, one event loop
//   track 4  QTSQ trajectory         getTrace(runId) replays every step
//   track 5  resilient editing       edit_file rewrites a per-run fixture
//
// The mock provider answers from the request it receives (RUN=<i> plus the
// tool messages so far), never from shared state, so any cross-run bleed
// shows up as one run's nonce inside another run's answer.

const NL = String.fromCharCode(10);
const RUNS = 50;

let passed = 0;
let failed = 0;
function check(cond, label) {
  if (cond) { passed++; console.log("  PASS " + label); }
  else { failed++; console.log("  FAIL " + label); }
}
function nonceOf(i) { return "N" + (1000 + i); }
function noncesIn(text) {
  const m = String(text || "").match(/N[0-9]{4}/g) || [];
  return m;
}

async function main() {
  const T0 = Date.now();
  console.log("=== Phase 6: 50-run modernization torture suite ===");

  /* ── Mock provider ─────────────────────────────────────────────── */
  let inFlight = 0;
  let maxInFlight = 0;
  let requests = 0;

  function sseOpen(res) { res.writeHead(200, { "Content-Type": "text/event-stream" }); }
  function sseData(res, obj) { res.write("data: " + JSON.stringify(obj) + "\n\n"); }
  function sseEnd(res) {
    sseData(res, { usage: { prompt_tokens: 30, completion_tokens: 8 } });
    res.write("data: [DONE]\n\n");
    res.end();
    inFlight--;
  }
  function toolCall(res, id, name, args) {
    sseOpen(res);
    sseData(res, {
      choices: [{ delta: { tool_calls: [{ index: 0, id: id, function: { name: name, arguments: JSON.stringify(args) } }] } }]
    });
    sseEnd(res);
  }

  const handler = async (req, res) => {
    requests++;
    inFlight++;
    if (inFlight > maxInFlight) maxInFlight = inFlight;
    /* Yield inside the handler: without it the server drains each request
     * synchronously and "overlap" is unmeasurable (single-threaded loop).
     * With it, 50 runs really do hold 50 streams open at once. */
    await sofuu.sleep(10);
    /* sofuu's response object is not an event emitter (no res.on), so the
     * in-flight count is drained in sseEnd - every reply path ends there. */

    let body = {};
    try { body = JSON.parse(req.body || "{}"); } catch (_) {}
    const msgs = body.messages || [];
    let blob = "";
    const toolContents = [];
    for (const m of msgs) {
      const c = typeof m.content === "string" ? m.content : "";
      blob += c + NL;
      if (m.role === "tool") toolContents.push(c);
    }
    const found = blob.match(/RUN=([0-9]+)/);
    if (!found) {
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "NO-RUN-MARKER" } }] });
      sseEnd(res);
      return;
    }
    const i = parseInt(found[1], 10);
    const nonce = nonceOf(i);

    if (toolContents.length === 0) {
      return toolCall(res, "ctx_" + i, "ctx_probe", { nonce: nonce });
    }
    if (toolContents.length === 1) {
      if (i % 2 === 0) {
        return toolCall(res, "code_" + i, "code_eval", { code: "var n = " + i + "; 'K' + n + 'Z'" });
      }
      return toolCall(res, "edit_" + i, "edit_file", {
        path: "torture_" + i + ".txt",
        old_string: "OLD-" + i,
        new_string: "NEW-" + i
      });
    }
    sseOpen(res);
    sseData(res, { choices: [{ delta: { content: "DONE-" + i + "|" + toolContents.join("|") } }] });
    sseEnd(res);
  };

  let server = null;
  let port = 0;
  const base = 23000 + (Date.now() % 3000);
  for (let i = 0; i < 12; i++) {
    const p = base + i * 13;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, "127.0.0.1");
      port = p;
      break;
    } catch (_) {
      server = null;
    }
  }
  if (server === null) { console.error("could not bind a mock port"); process.exit(1); }
  const MOCK = "http://127.0.0.1:" + port + "/v1/chat/completions";
  console.log("mock provider on port " + port);

  /* ── QTSQ storage + the agent under test ───────────────────────── */
  const work = "/tmp/sofuu_torture_" + Date.now();
  await sofuu.fs.mkdir(work, { recursive: true });
  const sdb = sofuu.storage.open(work + "/storage");

  let probeCalls = 0;
  sofuu.agent.define({
    name: "torture",
    system: "AGENT=torture Stress probe. Use the tools exactly as instructed.",
    toolsMode: "full",
    memory: "off",
    rlm: "off",
    provider: "openai",
    model: "mock",
    api_key: "x",
    base_url: MOCK,
    tools: [
      "code",
      {
        name: "ctx_probe",
        description: "Report this run's context nonce.",
        readOnly: true,
        parameters: {
          type: "object",
          properties: { nonce: { type: "string", description: "the nonce to echo" } },
          required: ["nonce"]
        },
        execute: async (args) => {
          probeCalls++;
          await sofuu.sleep(2);
          const cur = sofuu.agent.current();
          const hasCtx = !!(cur && cur.runId && String(cur.runId).length > 0);
          return "CTX-" + String((args && args.nonce) || "?") + "-" + (hasCtx ? "ctx" : "noctx");
        }
      }
    ]
  });

  /* 25 fixtures for the editing track (even runs sandbox, odd runs edit). */
  for (let i = 0; i < RUNS; i++) {
    if (i % 2 === 1) await sofuu.fs.writeFile("torture_" + i + ".txt", "OLD-" + i + NL);
  }

  /* ── 50 concurrent runs ─────────────────────────────────────────── */
  const items = [];
  for (let i = 0; i < RUNS; i++) {
    items.push({ agent: "torture", task: "RUN=" + i + " nonce=" + nonceOf(i) + " do the drill", opts: { storage: sdb } });
  }
  const results = await sofuu.agent.runMany(items, { concurrency: RUNS });
  const elapsed = Date.now() - T0;

  console.log("");
  console.log("Test 1: the swarm actually ran concurrently");
  check(results.length === RUNS, "50 results returned (" + results.length + ")");
  check(maxInFlight >= 8, "provider saw >= 8 overlapping streams (max " + maxInFlight + ")");
  check(requests >= RUNS * 2, "at least 2 provider calls per run (" + requests + " requests)");
  check(elapsed < 180000, "whole swarm finished in " + elapsed + " ms");

  console.log("Test 2: every run completed on its own trajectory");
  const runIds = [];
  let answered = 0;
  let stoppedClean = 0;
  for (let i = 0; i < RUNS; i++) {
    const r = results[i] || {};
    if (r.runId) runIds.push(String(r.runId));
    if (!r.stopped) stoppedClean++;
    if (String(r.answer || "").indexOf("DONE-" + i + "|") >= 0) answered++;
  }
  check(stoppedClean === RUNS, "50/50 runs stopped cleanly (" + stoppedClean + ")");
  check(answered === RUNS, "50/50 answers carry their own RUN id (" + answered + ")");
  check(new Set(runIds).size === RUNS, "50 distinct runIds (" + new Set(runIds).size + ")");

  console.log("Test 3: no cross-run bleed (track 2 + 3)");
  let isolated = 0;
  let ctxOk = 0;
  for (let i = 0; i < RUNS; i++) {
    const ans = String((results[i] || {}).answer || "");
    const seen = noncesIn(ans);
    const mine = nonceOf(i);
    const foreign = seen.filter(function (n) { return n !== mine; });
    if (foreign.length === 0 && seen.indexOf(mine) >= 0) isolated++;
    if (ans.indexOf("CTX-" + mine + "-ctx") >= 0) ctxOk++;
  }
  check(isolated === RUNS, "50/50 answers contain only their own nonce (" + isolated + ")");
  check(ctxOk === RUNS, "50/50 inline tools saw their run's context (" + ctxOk + ")");
  check(probeCalls === RUNS, "ctx_probe ran once per run (" + probeCalls + ")");

  console.log("Test 4: sandboxed computation (track 1)");
  let sandboxed = 0;
  for (let i = 0; i < RUNS; i += 2) {
    const ans = String((results[i] || {}).answer || "");
    if (ans.indexOf("K" + i + "Z") >= 0) sandboxed++;
  }
  check(sandboxed === RUNS / 2, "25/25 code_eval tokens came back (" + sandboxed + ")");

  console.log("Test 5: resilient editing under load (track 5)");
  let edited = 0;
  for (let i = 1; i < RUNS; i += 2) {
    const text = await sofuu.fs.readFile("torture_" + i + ".txt", "utf8");
    if (text.indexOf("NEW-" + i) >= 0 && text.indexOf("OLD-" + i) < 0) edited++;
  }
  check(edited === RUNS / 2, "25/25 fixtures rewritten correctly (" + edited + ")");

  console.log("Test 6: QTSQ trajectory for every run (track 4)");
  let traced = 0;
  let tooled = 0;
  let nonceInTrace = 0;
  for (let i = 0; i < RUNS; i++) {
    const ev = await sofuu.agent.getTrace(results[i].runId, { storage: sdb });
    if (Array.isArray(ev) && ev.length >= 4) traced++;
    const tools = (ev || []).filter(function (e) { return e.kind === "tool"; });
    if (tools.length === 2) tooled++;
    const blob = JSON.stringify(ev || []);
    if (blob.indexOf(nonceOf(i)) >= 0) nonceInTrace++;
  }
  check(traced === RUNS, "50/50 runs have >= 4 trace events (" + traced + ")");
  check(tooled === RUNS, "50/50 traces hold exactly 2 tool events (" + tooled + ")");
  check(nonceInTrace === RUNS, "50/50 traces carry their run's nonce (" + nonceInTrace + ")");

  console.log("");
  console.log("wall time: " + elapsed + " ms, requests: " + requests + ", max streams: " + maxInFlight);
  if (failed === 0) console.log("ALL PHASE 6 TORTURE TESTS PASSED (" + passed + " checks)");
  else console.log("PHASE 6 TORTURE TESTS FAILED: " + failed + " of " + (passed + failed));

  /* ── cleanup ───────────────────────────────────────────────────── */
  try { if (sdb && typeof sdb.close === "function") sdb.close(); } catch (_) {}
  try { if (typeof server.close === "function") server.close(); } catch (_) {}
  for (let i = 1; i < RUNS; i += 2) { try { await sofuu.fs.rm("torture_" + i + ".txt"); } catch (_) {} }
  try { await sofuu.fs.rm(work, { recursive: true }); } catch (_) {}
}

main()
  .then(function () { process.exit(failed === 0 ? 0 : 1); })
  .catch(function (err) {
    console.error("Test crashed:", (err && (err.stack || err.message)) || err);
    process.exit(1);
  });
