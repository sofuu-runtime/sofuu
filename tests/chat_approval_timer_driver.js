// tests/chat_approval_timer_driver.js — js-3 (AUDIT-2026-09-07) RED/GREEN
// driver, run by tests/chat_approval_timer_e2e.sh (which provides HOME +
// config + cwd = project). Drives the JS chat engine (sofuu.chat.submit —
// the desktop driver), NOT the TUI's Rust driver.
//
// js-3: requestApproval() arms a hard-ceiling setTimeout per approval and
// never clears it when the approval resolves early. The desktop sets
// toolTimeoutMs to 1h, so every gated tool call leaks a 1h timer. The
// runtime caps the timer registry at MAX_TIMERS = 1024
// (rt/timer.rs timer_register / js_set_timer "too many timers") — past
// the cap setTimeout THROWS.
//
// Scenario: the mock LLM returns 8 write_file tool_calls per round for
// 140 rounds (1120 approvals > 1024), then the plain sentinel. The driver
// resolves every approval from onEvent (allow, no always-pass) — but on a
// MICROTASK, not synchronously inside the emit: the desktop resolves via
// a later message dispatch, so at arm time the approval is still pending.
// With the leak, approval #1025 cannot arm its ceiling timer: the
// setTimeout inside requestApproval's Promise executor throws BEFORE any
// resolve → the approval promise REJECTS → gateTool's .then never runs →
// write_file never executes and the agent loop reports
// "tool error: too many timers (max 1024)". (Synchronous resolution
// would mask the failure: resolve() settles the promise first, the
// executor's later throw is swallowed, and the orphaned write still runs.)
// Post-fix, resolved approvals clearTimeout immediately, the registry
// stays near-empty, and all 1120 files land.

const PORT = parseInt(process.argv[process.argv.length - 1] || "18931", 10);
const ROUNDS = 140;        // 140 tool rounds...
const PER_ROUND = 8;       // ...x 8 gated write_file calls = 1120 approvals
const SENTINEL = "APPROVAL-TIMER-OK";

let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

function sse(res, frames) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  for (const f of frames) res.write("data: " + f + "\n\n");
  res.end();
}

let round = 0;
const server = sofuu.createServer(function (req, res) {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  const roles = [];
  for (const m of msgs) {
    roles.push(m.tool_calls && m.tool_calls.length ? "assistant+tc" : String(m.role));
  }
  round++;
  if (round <= ROUNDS) {
    const tcs = [];
    for (let i = 0; i < PER_ROUND; i++) {
      tcs.push({
        index: i,
        id: "call_" + round + "_" + i,
        type: "function",
        function: {
          name: "write_file",
          arguments: JSON.stringify({ path: "w/ap_" + round + "_" + i + ".txt", content: "x" }),
        },
      });
    }
    console.log("MOCK-AP: round " + round + " -> " + tcs.length + " tool_calls");
    sse(res, [
      JSON.stringify({ choices: [{ delta: { tool_calls: tcs } }] }),
      JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 8 } }),
      "[DONE]",
    ]);
    return;
  }
  console.log("MOCK-AP: round " + round + " -> sentinel (roles=" + roles.length + " msgs)");
  sse(res, [
    JSON.stringify({ choices: [{ delta: { content: SENTINEL } }] }),
    JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } }),
    "[DONE]",
  ]);
});

async function main() {
  /* Single bind: the endpoint in the harness config points at exactly
   * this port, so a retry on PORT+1 would silently desync the test. */
  server.listen(PORT, "127.0.0.1");
  console.log("MOCK-AP-READY " + PORT);
}

main().then(async () => {
  const proj = process.env.SOFUU_PROJECT || process.cwd();
  await sofuu.fs.mkdir("w", { recursive: true });

  let approvals = 0, resolvErrs = 0, toolErrs = 0;
  const init = sofuu.chat.init({
    host: 'ap', project: proj,
    permissions: 'prompt',
    toolTimeoutMs: 3600000,   // the desktop's human-scale 1h ceiling
  });
  if (!init.ok) { console.error("FAIL harness: chat.init failed " + JSON.stringify(init)); process.exit(1); }

  const events = [];
  let submitErr = null;
  let submitRes = null;
  try {
    submitRes = await sofuu.chat.submit("APPROVAL-FLOOD build the files", {
      onEvent: function (e) {
        events.push(e);
        if (e.kind === "approval_request") {
          approvals++;
          /* Microtask resolution — mirrors the desktop resolving on a
           * later dispatch, after requestApproval's executor has finished
           * arming the ceiling timer (a synchronous resolve inside the
           * emit would settle the promise BEFORE the setTimeout line and
           * mask the throw entirely). */
          Promise.resolve().then(function () {
            const r = sofuu.chat.resolveApproval(e.payload.id, true);
            if (!r || r.ok !== true) resolvErrs++;
          });
        }
        if (e.kind === "tool_result" && e.payload && e.payload.error) {
          toolErrs++;
          if (toolErrs <= 3) console.log("TOOL-ERR sample: " + e.payload.error);
        }
      },
    });
  } catch (e) { submitErr = e; }

  const dones = events.filter(e => e.kind === "done");
  if (submitErr) console.log("submit threw: " + String((submitErr && submitErr.message) || submitErr));

  check("submit resolved (turn terminated)", !submitErr && !!submitRes && submitRes.ok === true);
  check("exactly one done event", dones.length === 1);
  check("done.answer = " + SENTINEL,
        dones.length === 1 && dones[0].payload.answer === SENTINEL);
  check("all " + (ROUNDS * PER_ROUND) + " approvals requested and resolved ok",
        approvals === ROUNDS * PER_ROUND && resolvErrs === 0);
  check("zero tool errors (approval path never threw)", toolErrs === 0);

  let missing = 0, firstMissing = "";
  for (let r = 1; r <= ROUNDS && missing < 5; r++) {
    for (let i = 0; i < PER_ROUND; i++) {
      const p = "w/ap_" + r + "_" + i + ".txt";
      let ex = false;
      try { ex = await sofuu.fs.exists(p); } catch (e) {}
      if (!ex) { missing++; if (!firstMissing) firstMissing = p; }
    }
  }
  /* The full sweep (cheap exists calls) — count everything. */
  if (missing < 5) {
    for (let r = 1; r <= ROUNDS; r++) {
      for (let i = 0; i < PER_ROUND; i++) {
        let ex = false;
        try { ex = await sofuu.fs.exists("w/ap_" + r + "_" + i + ".txt"); } catch (e) {}
        if (!ex) missing++;
      }
    }
  }
  check("all " + (ROUNDS * PER_ROUND) + " files written" +
        (missing ? " (first missing: " + firstMissing + ")" : ""), missing === 0);

  console.log(bad === 0 ? "\nAPPROVAL-TIMER DRIVER: ALL PASSED"
                        : "\nAPPROVAL-TIMER DRIVER: " + bad + " FAILED");
  process.exit(bad === 0 ? 0 : 1);
}).catch(e => { console.error("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
