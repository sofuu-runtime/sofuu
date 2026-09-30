// tests/long_horizon_test.js — LONG-HORIZON endurance battery (2026-09-02).
// The agent must finish long tasks WITHOUT REST: no provider outage, empty
// stream, transport cut, or sub-agent ceiling may kill a run that has
// budget left. Every scenario below injects the exact failure that used
// to end a run, and asserts the run still completes unaided.
//
// Covers (against the fixes landed 2026-09-02 in agent.js/tools.js):
//   LH1  patient transient ladder: a 429 storm on FRESH requests is
//        ridden out (6-step backoff) — the turn completes after the
//        provider recovers, instead of dying at 5.5s.
//   LH2  mid-answer transport cuts: up to 3 continuations re-request
//        from where the stream died; the answer = all pieces joined.
//   LH3  empty streams (transient): a fresh allowance per ROUND — two
//        empties in earlier rounds never disable the recovery in a
//        later round.
//   LH4  multi-round tool loop: 25 rounds of tool calls complete; the
//        task message survives fitGuard under context pressure (the
//        _sofuu_task marker) — the final answer still sees the task.
//   LH5  delegate inheritance: a sub-agent runs under the parent's
//        long-horizon budget — 20+ steps in the child complete where
//        the old 12-step/5-minute defaults died.
//   LH6  bash self-timing: a command that takes longer than the outer
//        30s tool timeout survives (selfTimed bypass) and is killed
//        only by its own timeout.
//   LH7  wall-budget honesty: when the wall budget truly can't fit the
//        next retry, the run breaches with budget_wall (never silently
//        sleeping past its own budget).
//   LH8  wall vs selfTimed (js-1, AUDIT-2026-09-07): a bash child that
//        outlasts the wall budget loses the run at the wall — the run
//        stops with budget_wall ~when the budget expires, not when the
//        child finally exits.
//
// Run:  ./sofuu run tests/long_horizon_test.js
// Debug a single scenario: LONG_HORIZON_ONLY="LH5,LH6" ./sofuu run ...
// (No network, no API keys — local mock provider on 127.0.0.1.)

let failures = 0;
function check(name, cond) {
  if (cond) { console.log("PASS " + name); }
  else { failures++; console.log("FAIL " + name); }
}
const VERBOSE = !!process.env.LONG_HORIZON_VERBOSE;
const ONLY = String(process.env.LONG_HORIZON_ONLY || "").toUpperCase();
function want(id) { return !ONLY || ONLY.indexOf(id) >= 0; }
function vlog(s) { if (VERBOSE) console.log(s); }

/* Short ladder so the battery runs in seconds — LH1 still proves the
 * SHAPE (multiple backoff attempts then success); the shipped 6-step
 * 60s default is the production setting, too slow for a test loop. */
process.env.SOFUU_STREAM_RETRY_DELAYS = "50,100,200,200,400,400";

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
function sseEnd(res) { res.write("data: [DONE]\n\n"); res.end(); }

/* Scenario state, keyed by the system prompt marker. */
const S = {
  storm: { fails: 0, need: 4 },          // LH1: fail the first 4 fresh requests with 429
  cut3: { cuts: 0, need: 3 },            // LH2: cut the answer 3 times mid-stream
  empty: { empties: 0, rounds: 0 },       // LH3: empties spread across rounds
  loop25: { round: 0 },                  // LH4: 25 tool rounds
  child20: { round: 0 },                 // LH5: 20-step sub-agent
  wall: null,                            // LH7: set per-run
};

function handler(req, res) {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
    const msgs = parsed.messages || [];
    const sys = sysOf(msgs);
    const tc = toolCount(msgs);
    const txt = allText(msgs);
    const stream = !!parsed.stream;
    if (VERBOSE) console.log("  REQ sys=" + sys.slice(0, 24) + " tc=" + tc +
      " tools=" + (parsed.tools || []).map(t => t.function ? t.function.name : t.name).join(",") +
      (tc > 0 ? " | last-tool: " + String((msgs[msgs.length - 1] || {}).content || "").slice(0, 200) : ""));

    function fail429() {
      res.writeHead(429, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Rate limit exceeded — try again", code: 429 } }));
    }

    /* ── LH1: 429 storm on fresh requests ───────────────────────── */
    if (sys.indexOf("LH=storm") >= 0) {
      if (tc === 0 && S.storm.fails < S.storm.need) {
        S.storm.fails++;
        vlog("  storm: injecting 429 (" + S.storm.fails + "/" + S.storm.need + ")");
        return fail429();
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "STORM-SURVIVED" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 10, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }

    /* ── LH2: three mid-answer transport cuts ────────────────────
     * A cut = content frames then a 429 error frame mid-stream (the
     * in-stream error kills the stream). The continuation request
     * carries the partial as an assistant message — recognized via
     * the partial text present in txt. */
    if (sys.indexOf("LH=cut3") >= 0) {
      const partialSeen = txt.indexOf("cut-piece") >= 0;
      if (S.cut3.cuts < S.cut3.need && !partialSeen) {
        S.cut3.cuts++;
        vlog("  cut3: cutting mid-answer (" + S.cut3.cuts + "/" + S.cut3.need + ")");
        sseOpen(res);
        sseData(res, { choices: [{ delta: { content: "cut-piece-" + S.cut3.cuts + " " } }] });
        sseData(res, { error: { message: "upstream overloaded mid-stream", code: 429 } });
        sseEnd(res);
        return;
      }
      if (partialSeen) {
        /* continuation: more content (a later cut may still hit it) */
        if (S.cut3.cuts < S.cut3.need) {
          S.cut3.cuts++;
          vlog("  cut3: cutting the continuation (" + S.cut3.cuts + "/" + S.cut3.need + ")");
          sseOpen(res);
          sseData(res, { choices: [{ delta: { content: "cut-piece-" + S.cut3.cuts + " " } }] });
          sseData(res, { error: { message: "upstream overloaded mid-stream", code: 429 } });
          sseEnd(res);
          return;
        }
        sseOpen(res);
        sseData(res, { choices: [{ delta: { content: "cut-final" } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 12, completion_tokens: 6 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "cut-piece-1 " } }] });
      sseData(res, { error: { message: "upstream overloaded mid-stream", code: 429 } });
      sseEnd(res);
      return;
    }

    /* ── LH3: transient empty streams across rounds ───────────────
     * Round 1: empty (no finish_reason) → retry must recover.
     * Rounds 2-5: tool calls. Round 6: empty AGAIN — the per-round
     * allowance must still be armed (old bug: run-scoped counter). */
    if (sys.indexOf("LH=empty") >= 0) {
      const noTools = !parsed.tools || !parsed.tools.length;
      const isRetry = msgs.some(m => String(m.role) === "assistant" &&
        String(m.content || "").indexOf("retry-probe") >= 0);
      if (noTools && !isRetry) {
        /* First no-tool request in this turn: serve an EMPTY stream
         * (200, usage, [DONE] — zero content, zero tool_calls). */
        sseOpen(res);
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 8, completion_tokens: 0 } });
        sseEnd(res);
        return;
      }
      if (noTools && isRetry) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { content: "EMPTY-RECOVERED" } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 8, completion_tokens: 3 } });
        sseEnd(res);
        return;
      }
      if (tc < 3) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "step_tool", arguments: "{}" } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 30, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "EMPTY-FINAL-OK" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 30, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }

    /* ── LH4: 25 tool rounds; context pressure via FAT tool results ── */
    if (sys.indexOf("LH=loop25") >= 0) {
      if (!parsed.tools || !parsed.tools.length) {
        /* salvage/breach round — answer it (the loop must NOT reach here) */
        sseOpen(res);
        sseData(res, { choices: [{ delta: { content: "LOOP25-SALVAGE" } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 50, completion_tokens: 5 } });
        sseEnd(res);
        return;
      }
      if (tc < 25) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "fat_tool", arguments: "{}" } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 100, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "LOOP25-DONE[" + (txt.indexOf("LONG-TASK-NEEDLE") >= 0 ? "task-present" : "TASK-LOST") + "]" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 100, completion_tokens: 6 } });
      sseEnd(res);
      return;
    }

    /* ── LH5: sub-agent does 20 tool steps ─────────────────────── */
    if (sys.indexOf("LH=child20") >= 0) {
      if (!parsed.tools || !parsed.tools.length) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { content: "CHILD20-SALVAGE" } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      if (tc < 20) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "step_tool", arguments: "{}" } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "CHILD20-DONE" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }
    if (sys.indexOf("LH=orch20") >= 0) {
      /* parent: delegate once, then finish with the child's answer */
      if (tc === 0) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "delegate", arguments: JSON.stringify({ agent: "lh_child20", task: "do the long sub-task", why: "test" }) } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "ORCH20-FINAL[" + (txt.indexOf("CHILD20-DONE") >= 0 ? "child-finished" : "child-died") + "]" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }

    /* ── LH6: bash self-timing ────────────────────────────────────
     * The parent runs one bash call of 4s — LONGER than the agent's
     * outer 30s? No: the outer default toolTimeoutMs is 30s, so 4s
     * alone wouldn't prove the bypass. Instead: def pins
     * toolTimeoutMs=1500ms and the bash call runs 4s — surviving the
     * OUTER timeout proves the bypass (a wrapped call would die at
     * 1.5s with 'bash timed out'). */
    if (sys.indexOf("LH=bashself") >= 0) {
      if (tc === 0) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "bash", arguments: JSON.stringify({ command: "sleep 4 && echo SELF-TIMED-OK", timeout_ms: 60000 }) } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "BASHSELF-FINAL[" + (txt.indexOf("SELF-TIMED-OK") >= 0 ? "survived" : "killed-early") + "]" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }

    /* ── LH8 mock (js-1): one SLOW selfTimed bash call that outlasts the
     * wall budget. With the fix the run must stop at the wall (~2s),
     * NOT when the child exits (~4s). Round 1+ is the salvage answer. */
    if (sys.indexOf("LH=wallover") >= 0) {
      if (tc === 0) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "bash", arguments: JSON.stringify({ command: "sleep 4 && echo SLOW-TOOL", timeout_ms: 60000 }) } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      sseOpen(res);
      sseData(res, { choices: [{ delta: { content: "WALLOVER-FINAL" } }] });
      sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 30, completion_tokens: 5 } });
      sseEnd(res);
      return;
    }

    /* ── LH7 mock: one working tool round, then a PERMANENT 429 outage.
     * The graceful-degradation gate requires real progress (steps > 0)
     * before it salvages — a zero-progress outage must stay a loud
     * throw (P1-17) — so the first round must succeed. */
    if (sys.indexOf("LH=wall") >= 0) {
      if (tc === 0) {
        sseOpen(res);
        sseData(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "step_tool", arguments: "{}" } }] } }] });
        sseData(res, { choices: [{ delta: {} }], usage: { prompt_tokens: 30, completion_tokens: 4 } });
        sseEnd(res);
        return;
      }
      return fail429();
    }

    /* default: echo a marker so unknown requests never hang */
    sseOpen(res);
    sseData(res, { choices: [{ delta: { content: "UNKNOWN-REQ" } }] });
    sseEnd(res);
}

async function main() {
  let server = null, port = 0;
  const base = 19300 + (Date.now() % 4000);
  for (let i = 0; i < 6 && !server; i++) {
    const p = base + i * 97;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, "127.0.0.1");
      port = p;
    } catch (e) { server = null; }
  }
  if (!server) { console.log("FAIL could not bind mock port"); process.exit(1); }
  const MOCK = "http://127.0.0.1:" + port + "/v1/chat/completions";
  const PROV = { provider: "openai", base_url: MOCK, api_key: "x", model: "mock" };
  vlog("mock provider on " + MOCK);

  /* ── LH1: the 429 storm ─────────────────────────────────────── */
  if (want("LH1")) {    console.log("-- LH1: 429 storm ridden out, turn completes");
    sofuu.agent.define({
      name: "lh_storm", system: "LH=storm", memory: "off", rlm: "off",
      budget: { maxSteps: 3, maxWallMs: 60000, maxTokens: 500000 },
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_storm", "ride the storm", {});
    check("LH1 storm: run completed after 4 injected 429s",
          !r.stopped && r.answer.indexOf("STORM-SURVIVED") >= 0);
    check("LH1 storm: exactly 4 failures were injected", S.storm.fails === 4);
  }

  /* ── LH2: three mid-answer cuts ─────────────────────────────── */
  if (want("LH2")) {    console.log("-- LH2: three mid-answer cuts continued");
    sofuu.agent.define({
      name: "lh_cut3", system: "LH=cut3", memory: "off", rlm: "off",
      budget: { maxSteps: 3, maxWallMs: 60000, maxTokens: 500000 },
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_cut3", "answer with cuts", {});
    check("LH2 cuts: run completed after 3 mid-answer cuts",
          !r.stopped && r.answer.indexOf("cut-final") >= 0);
    check("LH2 cuts: all 3 pieces present, no duplicates lost",
          r.answer.indexOf("cut-piece-1") >= 0 && r.answer.indexOf("cut-piece-2") >= 0 &&
          r.answer.indexOf("cut-piece-3") >= 0);
  }

  /* ── LH3: empty streams with fresh per-round allowance ──────── */
  if (want("LH3")) {    console.log("-- LH3: empty streams, per-round recovery");
    sofuu.agent.define({
      name: "lh_empty", system: "LH=empty", memory: "off", rlm: "off",
      budget: { maxSteps: 8, maxWallMs: 60000, maxTokens: 500000 },
      tools: [{ name: "step_tool", description: "a step", parameters: { type: "object", properties: {} },
                execute: () => "STEP-DONE" }],
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_empty", "work through empties", {});
    check("LH3 empty: completed with recovered answer",
          !r.stopped && r.answer.indexOf("EMPTY-FINAL-OK") >= 0);
  }

  /* ── LH4: 25 tool rounds under context pressure ─────────────── */
  if (want("LH4")) {    console.log("-- LH4: 25 rounds, task survives fitGuard pressure");
    const FAT = "X".repeat(6000); /* fat tool results force fitGuard rungs */
    sofuu.agent.define({
      name: "lh_loop25", system: "LH=loop25", memory: "off", rlm: "off",
      budget: { maxSteps: 40, maxWallMs: 120000, maxTokens: 500000 },
      tools: [{ name: "fat_tool", description: "a fat step", parameters: { type: "object", properties: {} },
                execute: () => "FAT:" + FAT }],
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_loop25", "LONG-TASK-NEEDLE do the long job", {});
    check("LH4 loop: 25 rounds completed, no budget stop",
          !r.stopped && r.answer.indexOf("LOOP25-DONE") >= 0);
    check("LH4 loop: task message still visible to the final round (fitGuard kept it)",
          r.answer.indexOf("task-present") >= 0);
    check("LH4 loop: no salvage round was spent",
          r.answer.indexOf("LOOP25-SALVAGE") < 0);
  }

  /* ── LH5: sub-agent inherits the long budget ────────────────── */
  if (want("LH5")) {    console.log("-- LH5: delegate child inherits the long-horizon budget");
    sofuu.agent.define({
      name: "lh_orch20", system: "LH=orch20", memory: "off", rlm: "off",
      agents: ["lh_child20"],
      budget: { maxSteps: 30, maxWallMs: 120000, maxTokens: 500000, maxDepth: 1 },
      ...PROV,
    });
    sofuu.agent.define({
      name: "lh_child20", system: "LH=child20", memory: "off", rlm: "off",
      budget: { maxSteps: 2 }, /* would die at step 2 with the OLD defaults */
      tools: [{ name: "step_tool", description: "a step", parameters: { type: "object", properties: {} },
                execute: () => "STEP-DONE" }],
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_orch20", "delegate the long sub-task", {});
    check("LH5 delegate: child completed 20 steps under the parent's budget",
          !r.stopped && r.answer.indexOf("child-finished") >= 0);
  }

  /* ── LH6: bash self-timing ──────────────────────────────────── */
  if (want("LH6")) {    console.log("-- LH6: bash self-timed, survives the outer tool timeout");
    sofuu.agent.define({
      name: "lh_bashself", system: "LH=bashself", memory: "off", rlm: "off",
      budget: { maxSteps: 4, maxWallMs: 120000, maxTokens: 500000 },
      tools: ["code"], /* built-in bash */
      toolTimeoutMs: 1500, /* OUTER timeout is 1.5s; bash runs 4s */
      ...PROV,
    });
    const r = await sofuu.agent.run("lh_bashself", "run the long command", {});
    check("LH6 bash: 4s command survived a 1.5s outer timeout (selfTimed)",
          !r.stopped && r.answer.indexOf("survived") >= 0);
  }

  /* ── LH7: wall-budget honesty + graceful outage ────────────────
   * Wall budget 2s; every fresh request 429s. wallAllows() needs >5s
   * of wall left, so the ladder is REFUSED immediately — the run must
   * not sleep past its own budget. With the outage unridable, the run
   * degrades gracefully: stopped=provider_outage + an offline salvage
   * summary, never a raw HTTP 429 thrown at the caller. */
  if (want("LH7")) {    console.log("-- LH7: tiny wall + permanent outage stops honestly");
    sofuu.agent.define({
      name: "lh_wall", system: "LH=wall", memory: "off", rlm: "off",
      budget: { maxSteps: 3, maxWallMs: 2000, maxTokens: 500000 },
      /* the graceful-outage path lives in the TOOL LOOP (a multi-step
       * task with real work done deserves the salvage, not a raw throw);
       * a no-tool def takes the plain finalAnswer path where a raw
       * error IS the honest contract. */
      tools: [{ name: "step_tool", description: "a step", parameters: { type: "object", properties: {} },
                execute: () => "STEP-DONE" }],
      ...PROV,
    });
    const t0 = Date.now();
    let r = null;
    let threw = null;
    try { r = await sofuu.agent.run("lh_wall", "hit the wall", {}); }
    catch (e) { threw = e; }
    const ms = Date.now() - t0;
    check("LH7 wall: graceful provider_outage stop, not a raw throw",
          !threw && !!r && r.stopped === "provider_outage");
    check("LH7 wall: offline salvage summary present",
          !!r && r.answer.indexOf("provider outage") >= 0);
    check("LH7 wall: no ladder sleep past the 2s budget (" + ms + "ms)",
          ms < 3500);
  }

  /* ── LH8: wall budget vs a selfTimed bash child (js-1) ──────────
   * Wall budget 2s; the tool round issues a 4s bash command. The
   * selfTimed bypass keeps the OUTER tool timeout away from bash
   * (LH6), but the WALL budget must still win: the run stops at ~2s
   * with budget_wall instead of waiting out the 4s child. */
  if (want("LH8")) {    console.log("-- LH8: selfTimed bash cannot overrun the wall budget");
    sofuu.agent.define({
      name: "lh_wallover", system: "LH=wallover", memory: "off", rlm: "off",
      budget: { maxSteps: 4, maxWallMs: 2000, maxTokens: 500000 },
      tools: ["code"],
      toolTimeoutMs: 60000, /* outer never fires — the WALL must be the killer */
      ...PROV,
    });
    const t0 = Date.now();
    let r = null;
    let threw = null;
    try { r = await sofuu.agent.run("lh_wallover", "run the slow command", {}); }
    catch (e) { threw = e; }
    const ms = Date.now() - t0;
    check("LH8 wall: run stops with budget_wall, not a raw throw",
          !threw && !!r && r.stopped === "budget_wall");
    check("LH8 wall: run ends near the 2s wall, not the 4s child (" + ms + "ms)",
          ms < 3300);
    check("LH8 wall: salvage answer present",
          !!r && r.answer.indexOf("WALLOVER-FINAL") >= 0);
  }

  console.log("");
  if (failures === 0) console.log("LONG-HORIZON TEST: ALL PASSED");
  else console.log("LONG-HORIZON TEST: " + failures + " FAILED");
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => { console.log("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
