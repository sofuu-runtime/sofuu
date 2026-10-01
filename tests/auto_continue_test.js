/* ========================================================================
 * tests/auto_continue_test.js — the step budget is a guardrail, not a wall.
 *
 * Before 2026-10-01 a run that used its step allowance salvaged one summary
 * and stopped, so a genuinely long task ended half-finished with only
 * "step budget reached" to show. Now the run compacts its live transcript
 * and renews the allowance, bounded by maxContinuations.
 *
 * The mock here is a long-but-CONVERGING task: it grinds through many
 * distinct tool calls and then finishes. That is the case the budget
 * exists to serve, and it must now survive the trip.
 * ======================================================================== */

let port = 0;
let round = 0;
let spinMode = false;          /* never converge: tool call forever */
const DISTINCT_CALLS = 6;      /* rounds of unique work, then done */
const BIG_RESULT = "X".repeat(900); /* fat results => transcript grows */

function handler(req, res) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  if (!bodyHasTools(req)) {
    /* Salvage round: answer in plain text. */
    text(res, "SALVAGE");
    return;
  }
  round++;
  if (spinMode) {
    tool(res, "list_dir", { path: "spin-" + round }, "S" + BIG_RESULT);
  } else if (round <= DISTINCT_CALLS) {
    /* Unique args each round, and a fat result, so the live transcript
     * grows past the compaction threshold. */
    tool(res, "list_dir", { path: "dir-" + round }, "R" + round + BIG_RESULT);
  } else {
    text(res, "FINISHED-AFTER-CONTINUE");
  }
}

/* req.body arrives as a raw string — parse before inspecting. */
function reqJSON(req) {
  try {
    const b = req.body;
    if (!b) return {};
    return typeof b === "string" ? JSON.parse(b) : b;
  } catch (e) { return {}; }
}
function bodyHasTools(req) {
  const b = reqJSON(req);
  return !!(b.tools && b.tools.length);
}
function text(res, s) {
  const c = { id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: { content: s }, finish_reason: null }] };
  res.write("data: " + JSON.stringify(c) + "\n\n");
  const f = { id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }] };
  res.write("data: " + JSON.stringify(f) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
}
function tool(res, name, args, content) {
  const c = { id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "c" + Math.random().toString(36).slice(2),
      type: "function", function: { name: name, arguments: JSON.stringify(args) } }] }, finish_reason: null }] };
  res.write("data: " + JSON.stringify(c) + "\n\n");
  const f = { id: "x", object: "chat.completion.chunk", model: "m",
    choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] };
  res.write("data: " + JSON.stringify(f) + "\n\n");
  res.write("data: [DONE]\n\n");
  res.end();
  void content;
}

const results = { total: 0, passed: 0, failed: 0 };
function check(label, ok, detail) {
  console.log((ok ? "PASS " : "FAIL ") + label + (detail ? "  [" + detail + "]" : ""));
  results.total++;
  if (ok) results.passed++; else results.failed++;
}

async function main() {
  let server = null;
  const base = 28400 + (Date.now() % 700);
  for (let i = 0; i < 8 && !server; i++) {
    try {
      server = sofuu.createServer(handler);
      server.listen(base + i * 31, "127.0.0.1");
      port = base + i * 31;
    } catch (e) { server = null; }
  }
  if (!server) { console.log("FAIL could not bind mock port"); process.exit(1); }
  const BASE = "http://127.0.0.1:" + port;

  /* maxSteps 3 with 6 distinct rounds: cannot finish in one window, so a
   * correct runtime MUST renew rather than stop. */
  sofuu.agent.define({
    name: "longer", system: "You grind through distinct tool calls then finish.",
    tools: [{ name: "list_dir", description: "list",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      execute: async (a) => "R" + a.path + "X".repeat(900) }],
    memory: "off", rlm: "off",
    budget: { maxSteps: 3, maxContinuations: 2 },
    provider: "openai", model: "mock", api_key: "x", base_url: BASE,
  });

  const r = await sofuu.agent.run("longer", "do the long grind", {});

  check("the run finished instead of stopping", r.answer.indexOf("FINISHED-AFTER-CONTINUE") >= 0,
        JSON.stringify((r.answer || "").slice(0, 50)));
  check("not stopped at the step budget", !r.stopped, "stopped=" + r.stopped);
  check("cumulative steps exceed one window", r.steps > 3, "steps=" + r.steps);

  /* The transcript must have been compacted, not just the counter reset —
   * otherwise the next window re-sends the same growing transcript and
   * the window is the next thing to blow. */
  const kinds = r.trace.map((e) => e.kind);
  check("a compaction/renewal event was emitted", kinds.indexOf("plan") >= 0);
  const warned = r.trace.filter((e) => e.kind === "warn");
  check("the user was told it continued", warned.length > 0);

  /* Renewal must be BOUNDED: a model that never converges still stops. */
  spinMode = true; round = 0; /* never converge */
  sofuu.agent.define({
    name: "spins", system: "AGENT=never-converges",
    tools: [{ name: "list_dir", description: "list",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      execute: async (a) => "R" + a.path + "X".repeat(900) }],
    memory: "off", rlm: "off",
    budget: { maxSteps: 3, maxContinuations: 2 },
    provider: "openai", model: "mock", api_key: "x", base_url: BASE,
  });
  const r2 = await sofuu.agent.run("spins", "never finish", {});
  check("a non-converging run still stops", !!r2.stopped, "stopped=" + r2.stopped);
  check("it stops on the step budget, not something else",
        r2.stopped === "budget_steps", "stopped=" + r2.stopped);
  if (process.env.ACDBG) console.log("[r2] breach=" + JSON.stringify(r2.breach) +
    " llm=" + r2.usage.llmCalls + " steps=" + r2.steps);
  check("renewals were bounded (3 windows x maxSteps 3)",
        r2.steps === 9, "steps=" + r2.steps);
  check("the final stop salvages an answer",
        typeof r2.answer === "string" && r2.answer.length > 0);

  /* Continuations must be disableable (0 restores the old hard stop). */
  sofuu.agent.define({
    name: "nocont", system: "AGENT=never-converges",
    tools: [{ name: "list_dir", description: "list",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      execute: async (a) => "R" + a.path }],
    memory: "off", rlm: "off",
    budget: { maxSteps: 2, maxContinuations: 0 },
    provider: "openai", model: "mock", api_key: "x", base_url: BASE,
  });
  const r3 = await sofuu.agent.run("nocont", "hard stop please", {});
  check("maxContinuations:0 restores the hard stop",
        r3.stopped === "budget_steps" && r3.steps === 2,
        "stopped=" + r3.stopped + " steps=" + r3.steps);

  console.log("");
  console.log("=== RESULTS ===");
  console.log("Passed: " + results.passed + " | Failed: " + results.failed);
  if (results.failed > 0) process.exit(1);
  console.log("");
  console.log("✅ Auto-continue works");
}

main().catch((e) => {
  console.error("Test failed:", e && e.message ? e.message : e);
  process.exit(1);
});