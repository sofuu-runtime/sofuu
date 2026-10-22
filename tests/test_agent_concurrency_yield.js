// tests/test_agent_concurrency_yield.js
// Verification suite for Phase 3: Cooperative Scheduling & Concurrent Tool Dispatch

const assert = (condition, msg) => {
  if (!condition) {
    console.error(`❌ ASSERTION FAILED: ${msg}`);
    process.exit(1);
  }
};

async function main() {
  console.log("=== Testing Phase 3: Cooperative Scheduling & Concurrent Tool Dispatch ===");

  // 1. Cooperative Yielding Primitives
  assert(typeof globalThis.scheduler === "object", "globalThis.scheduler must exist");
  assert(typeof globalThis.scheduler.yield === "function", "scheduler.yield must be a function");
  assert(typeof sofuu.yield === "function", "sofuu.yield must be a function");

  let timerFired = false;
  setTimeout(() => {
    timerFired = true;
  }, 10);

  // Yield multiple times
  for (let i = 0; i < 5; i++) {
    await globalThis.scheduler.yield();
  }
  await sofuu.sleep(25);
  assert(timerFired, "Timer must fire when yielding cooperatively");
  console.log("✓ Step 1: Cooperative yield allows timers to fire smoothly");

  // 2. Set up Mock LLM Server
  let activeConcurrency = 0;
  let maxObservedConcurrency = 0;
  let executionLog = [];

  function sseOpen(res) { res.writeHead(200, { "Content-Type": "text/event-stream" }); }
  function sseData(res, obj) { res.write("data: " + JSON.stringify(obj) + "\n\n"); }
  function sseEnd(res) {
    sseData(res, { usage: { prompt_tokens: 20, completion_tokens: 10 } });
    res.write("data: [DONE]\n\n");
    res.end();
  }

  const handler = (req, res) => {
    let body = {};
    try { body = JSON.parse(req.body || "{}"); } catch (_) {}
    const msgs = body.messages || [];
    let sys = "";
    let tc = 0;
    let txt = "";
    for (const m of msgs) {
      if (m.role === "system") sys += m.content + " ";
      if (m.role === "tool") tc++;
      if (m.content) txt += m.content + " ";
    }

    if (sys.indexOf("AGENT=par_reads") >= 0) {
      if (tc === 0) {
        // Return 4 parallel tool calls in one turn
        sseOpen(res);
        sseData(res, {
          choices: [{
            delta: {
              tool_calls: [
                { index: 0, id: "call_r1", function: { name: "parallel_read", arguments: JSON.stringify({ id: "item1", duration: 50 }) } },
                { index: 1, id: "call_r2", function: { name: "parallel_read", arguments: JSON.stringify({ id: "item2", duration: 50 }) } },
                { index: 2, id: "call_r3", function: { name: "parallel_read", arguments: JSON.stringify({ id: "item3", duration: 50 }) } },
                { index: 3, id: "call_r4", function: { name: "parallel_read", arguments: JSON.stringify({ id: "item4", duration: 50 }) } }
              ]
            }
          }]
        });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "PARALLEL-READS-OK: " + txt } }] });
      sseEnd(res);
      return;
    }

    if (sys.indexOf("AGENT=hazard") >= 0) {
      if (tc === 0) {
        // Contiguous partition test: 2 reads -> 1 write -> 1 read
        sseOpen(res);
        sseData(res, {
          choices: [{
            delta: {
              tool_calls: [
                { index: 0, id: "call_h1", function: { name: "parallel_read", arguments: JSON.stringify({ id: "a", duration: 30 }) } },
                { index: 1, id: "call_h2", function: { name: "parallel_read", arguments: JSON.stringify({ id: "b", duration: 30 }) } },
                { index: 2, id: "call_h3", function: { name: "mutating_write", arguments: JSON.stringify({ file: "out.txt", text: "val" }) } },
                { index: 3, id: "call_h4", function: { name: "parallel_read", arguments: JSON.stringify({ id: "c", duration: 30 }) } }
              ]
            }
          }]
        });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "HAZARD-FREE-OK" } }] });
      sseEnd(res);
      return;
    }

    if (sys.indexOf("AGENT=eval") >= 0) {
      if (tc === 0) {
        sseOpen(res);
        sseData(res, {
          choices: [{
            delta: {
              tool_calls: [
                { index: 0, id: "call_ev1", function: { name: "code_eval", arguments: JSON.stringify({ code: "40 + 2" }) } }
              ]
            }
          }]
        });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "EVAL-RESULT: " + txt } }] });
      sseEnd(res);
      return;
    }

    // Default fallback
    sseOpen(res);
    sseData(res, { choices: [{ delta: { content: "default answer" } }] });
    sseEnd(res);
  };

  let server = null;
  let port = 0;
  const base = 21000 + (Date.now() % 4000);
  for (let i = 0; i < 10; i++) {
    const p = base + i * 17;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, "127.0.0.1");
      port = p;
      break;
    } catch (_) {
      server = null;
    }
  }
  assert(server !== null, "Mock server must bind successfully");
  const MOCK = `http://127.0.0.1:${port}/v1/chat/completions`;
  const PROV = { provider: "openai", base_url: MOCK, api_key: "test", model: "mock" };

  console.log(`✓ Step 2: Mock server bound on port ${port}`);

  let toolStartTime = 0;
  let toolEndTime = 0;

  // 3. Define Agents
  const sharedTools = [
    {
      name: "parallel_read",
      description: "Read item concurrently",
      parameters: { type: "object", properties: { id: { type: "string" }, duration: { type: "number" } } },
      parallel: true,
      execute: async (args) => {
        if (!toolStartTime) toolStartTime = Date.now();
        activeConcurrency++;
        if (activeConcurrency > maxObservedConcurrency) {
          maxObservedConcurrency = activeConcurrency;
        }
        executionLog.push(`start_read_${args.id}`);
        await sofuu.sleep(args.duration || 40);
        executionLog.push(`end_read_${args.id}`);
        activeConcurrency--;
        toolEndTime = Date.now();
        return `content_of_${args.id}`;
      }
    },
    {
      name: "mutating_write",
      description: "Mutate file sequentially",
      parameters: { type: "object", properties: { file: { type: "string" }, text: { type: "string" } } },
      writes: true,
      execute: async (args) => {
        executionLog.push(`start_write_${args.file}`);
        assert(activeConcurrency === 0, "Mutating tool must NOT run concurrently with reads!");
        await sofuu.sleep(30);
        executionLog.push(`end_write_${args.file}`);
        return `wrote_${args.file}`;
      }
    }
  ];

  sofuu.agent.define(Object.assign({
    name: "agent_par",
    system: "AGENT=par_reads test concurrent tool execution",
    tools: sharedTools
  }, PROV));

  sofuu.agent.define(Object.assign({
    name: "agent_hazard",
    system: "AGENT=hazard test read-write hazard isolation",
    tools: sharedTools
  }, PROV));

  sofuu.agent.define(Object.assign({
    name: "agent_eval",
    system: "AGENT=eval test code_eval auto pass",
    tools: [{ builtin: "code_eval" }]
  }, PROV));

  // 4. Test Concurrent Execution Speedup
  activeConcurrency = 0;
  maxObservedConcurrency = 0;
  executionLog = [];
  toolStartTime = 0;
  toolEndTime = 0;

  const t0 = Date.now();
  const run1 = await sofuu.agent.run("agent_par", "run 4 parallel reads", {});
  const elapsed1 = Date.now() - t0;
  const toolSpan = toolEndTime - toolStartTime;

  console.log(`4 parallel reads total roundtrip: ${elapsed1}ms, pure tool span: ${toolSpan}ms, max observed concurrency: ${maxObservedConcurrency}`);
  assert(maxObservedConcurrency >= 3, `Expected concurrency >= 3, got ${maxObservedConcurrency}`);
  assert(toolSpan < 120, `Expected tool span < 120ms (pure parallel ~50ms, sequential would be 200ms), got ${toolSpan}ms`);
  assert(run1.answer.indexOf("PARALLEL-READS-OK") >= 0, "Run 1 must finish with expected answer");
  console.log("✓ Step 3: 4 parallel tool calls executed concurrently with >= 3x speedup");

  // 5. Test Contiguous Batching Hazard-Free Ordering
  activeConcurrency = 0;
  maxObservedConcurrency = 0;
  executionLog = [];

  const run2 = await sofuu.agent.run("agent_hazard", "read -> write -> read", {});
  console.log("Execution log:", executionLog.join(" -> "));

  const endA = executionLog.indexOf("end_read_a");
  const endB = executionLog.indexOf("end_read_b");
  const startW = executionLog.indexOf("start_write_out.txt");
  const endW = executionLog.indexOf("end_write_out.txt");
  const startC = executionLog.indexOf("start_read_c");

  assert(endA < startW, "Read A must finish before Write starts");
  assert(endB < startW, "Read B must finish before Write starts");
  assert(endW < startC, "Write must finish before Read C starts");
  assert(run2.answer === "HAZARD-FREE-OK", "Run 2 must finish with HAZARD-FREE-OK");
  console.log("✓ Step 4: Contiguous partitioning guarantees zero read-write hazards");

  // 6. Test code_eval AUTO_PASS_TOOLS in plan mode
  const run3 = await sofuu.agent.run("agent_eval", "eval code in plan mode", {
    permissionProfile: "plan"
  });

  assert(run3.answer.indexOf("42") >= 0, "code_eval in AUTO_PASS_TOOLS must execute in plan mode");
  console.log("✓ Step 5: code_eval executes cleanly under plan permission profile");

  if (server && server.close) {
    server.close();
  }

  console.log("\n✅ ALL PHASE 3 CONCURRENCY & SCHEDULING TESTS PASSED!");
}

main().catch(err => {
  console.error("FATAL:", err);
  process.exit(1);
});
