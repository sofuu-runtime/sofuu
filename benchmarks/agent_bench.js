// benchmarks/agent_bench.js — agent orchestration bench.
//
// Part 1 (legacy, QTSQ builds): CMA+KV prefetch orchestration demo via
//          sofuu.agent.create (memory.rs surface).
// Part 2 (PLAN-AGENTS A9.3): agents-vs-single-loop step latency — the same
//          scripted tool-rounds driven through (a) sofuu.agent.run and
//          (b) a hand-rolled raw ai.stream loop, against a local mock
//          provider with a fixed per-request delay. The delta is the
//          per-step overhead the agent layer ADDS (tool resolution, budget
//          checks, trace events) — the regression signal. Also proves the
//          registry survives a define+run+run cycle (run-twice).
//
// Run:  ./sofuu run benchmarks/agent_bench.js

/* ── Part 1: legacy CMA+KV prefetch demo ─────────────────────────── */
try {
    console.log("== Agent Orchestration Test ==");

    const dim = 16;
    let mem = Sofuu.memory.open("agent_mind.qtsq", dim);

    let kv = Sofuu.kv.open("agent_kv", {
        modelId: "test", nLayers: 2, nHeads: 4, headDim: 32
    });

    // Create an agent tying CMA and KV together
    let agent = Sofuu.agent.create("Alpha", mem, kv);
    console.log("Agent 'Alpha' successfully created uniting CMA and KV Store.");

    // Populate memories and KV cache
    const kData = new Float32Array(2*4*10*32);
    const vData = new Float32Array(2*4*10*32);
    const pageId = kv.save(kData, vData, 10);

    const contextVec = new Float32Array(dim);
    contextVec[1] = 1.0;

    // Remember a fact and link it to the exact KV page
    mem.remember(contextVec, "User prefers dark mode.", "system", pageId);
    console.log(`Saved fact into Cognitive Memory linking to KV Page ID ${pageId}`);

    console.log("Agent orchestrating prefetch routine based on a similar inference query...");
    const query = new Float32Array(dim);
    query[1] = 0.99; // very similar

    // Instruct agent to prefetch
    const numPrefetched = agent.prefetch(query, 5);
    console.log("Agent Prefetch successful.");
    console.log(`Agent loaded ${numPrefetched} specific KV sequence blocks into ultra-low latency RAM from SSD based on Cognitive Similarity!`);

    console.log("== Test Complete ==");
} catch(e) { console.log("(legacy CMA+KV section skipped: " + (e && e.message || e) + ")"); }

/* ── Part 2: agents-vs-single-loop step latency (A9.3) ───────────── */

const MOCK_DELAY_MS = 8;   // artificial per-request latency in the mock
const ROUNDS = 6;          // tool rounds per measured run
const REPETITIONS = 3;     // measured repetitions per shape

async function bench() {
  /* Mock OpenAI-compatible provider: answers a tool call until it has
   * seen ROUNDS tool results, then answers text (each measured run =
   * ROUNDS tool rounds + one final). */
  const srv = sofuu.createServer((req, res) => {
    let body = {};
    try { body = JSON.parse(req.body || "{}"); } catch (e) {}
    const msgs = body.messages || [];
    let toolCount = 0;
    for (const m of msgs) if (String(m.role) === "tool") toolCount++;
    setTimeout(() => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      const ev = (o) => res.write("data: " + JSON.stringify(o) + "\n\n");
      if (toolCount < ROUNDS) {
        ev({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "noop", arguments: "{}" } }] } }] });
      } else {
        ev({ choices: [{ delta: { content: "done" } }] });
      }
      ev({ usage: { prompt_tokens: 40, completion_tokens: 8 } });
      res.write("data: [DONE]\n\n");
      res.end();
    }, MOCK_DELAY_MS);
  });

  let port = 0;
  for (let i = 0; i < 8; i++) {
    try { srv.listen(21600 + i * 13, "127.0.0.1"); port = 21600 + i * 13; break; } catch (e) {}
  }
  if (!port) { console.log("FAIL could not bind mock"); process.exit(1); }
  const MOCK = "http://127.0.0.1:" + port + "/v1/chat/completions";

  sofuu.agent.define({
    name: "bencher", system: "bench agent", memory: "off", rlm: "off",
    tools: [{ name: "noop", description: "n", parameters: { type: "object", properties: {} },
              execute: async () => "ok" }],
    provider: "openai", model: "bench", api_key: "x", base_url: MOCK,
  });

  /* (a) agent loop: ROUNDS tool rounds + final, per run */
  const agentTimes = [];
  for (let r = 0; r < REPETITIONS; r++) {
    const t0 = Date.now();
    const res = await sofuu.agent.run("bencher", "bench run " + r,
      { budget: { maxSteps: ROUNDS + 1, maxDepth: 1, maxTokens: 10000000, maxWallMs: 120000 } });
    if (res.stopped) { console.log("FAIL agent bench stopped: " + res.stopped); process.exit(1); }
    agentTimes.push(Date.now() - t0);
  }

  /* (b) single loop: the SAME wire pattern, hand-rolled — the baseline the
   * agent layer is compared against (one stream per round, tools executed
   * inline, no registry/trace/budget machinery). */
  async function singleRun() {
    const t0 = Date.now();
    const msgs = [{ role: "user", content: "bench" }];
    const loopMsgs = [];
    for (let step = 0; step <= ROUNDS; step++) {
      const st = sofuu.ai.stream({
        messages: msgs.concat(loopMsgs),
        tools: [{ name: "noop", description: "n", parameters: { type: "object", properties: {} } }],
        provider: "openai", model: "bench", api_key: "x", base_url: MOCK,
      });
      for await (const c of st) { /* drain */ }
      if (!(st.toolCalls || []).length) break;
      loopMsgs.push({ role: "assistant", content: null,
        tool_calls: [{ id: "c1", type: "function", function: { name: "noop", arguments: "{}" } }] });
      loopMsgs.push({ role: "tool", tool_call_id: "c1", content: "ok" });
    }
    return Date.now() - t0;
  }
  const singleTimes = [];
  for (let r = 0; r < REPETITIONS; r++) singleTimes.push(await singleRun());

  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const aAvg = avg(agentTimes), sAvg = avg(singleTimes);
  const perStep = (aAvg - sAvg) / ROUNDS;

  console.log("\n== Agents-vs-single-loop step latency (A9.3) ==");
  console.log("  mock delay " + MOCK_DELAY_MS + "ms/req · " + ROUNDS + " tool rounds · " + REPETITIONS + " reps");
  console.log("  agent.run   : " + agentTimes.join(" / ") + " ms  (avg " + aAvg.toFixed(1) + ")");
  console.log("  single loop : " + singleTimes.join(" / ") + " ms  (avg " + sAvg.toFixed(1) + ")");
  console.log("  agent-layer overhead: +" + perStep.toFixed(2) + " ms per step");

  /* Registry pollution guard (PLAN-AGENTS risk table): define+run+run
   * leaves exactly one registry entry and clean run tables. */
  const before = sofuu.agent.list().length;
  await sofuu.agent.run("bencher", "pollution check 1", { budget: { maxSteps: 2, maxTokens: 10000000 } });
  await sofuu.agent.run("bencher", "pollution check 2", { budget: { maxSteps: 2, maxTokens: 10000000 } });
  const after = sofuu.agent.list().length;
  console.log("  run-twice registry: " + before + " → " + after + (before === after ? " (stable)" : " (LEAK!)"));

  process.exit(0);
}

(async function () {
  if (typeof sofuu !== "undefined" && sofuu.agent && sofuu.agent.run) {
    await bench();
  } else {
    console.log("sofuu.agent unavailable — skipping A9.3 bench");
  }
})().catch(e => { console.error("BENCH FAIL: " + (e && e.stack || e)); process.exit(1); });

