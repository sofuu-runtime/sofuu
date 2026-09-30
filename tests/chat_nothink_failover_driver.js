// tests/chat_nothink_failover_driver.js — js-2 (AUDIT-2026-09-07) RED/GREEN
// driver, run by tests/chat_nothink_failover_e2e.sh (which provides the
// HOME + config). Drives the JS chat engine (sofuu.chat.submit — the
// desktop driver), NOT the TUI's Rust driver.
//
// Scenario: the ACTIVE provider (mock-active) rejects ask 1 with
// HTTP 400 "reasoning_effort is not supported" → chat.js's noThink
// branch fires and retries the SAME provider WITHOUT the effort param.
// The retry is answered HTTP 429 (capacity-class). The failover chain's
// second provider (custom2/mock-failover) serves the plain answer.
//
// With the js-2 bug, the retry's 429 throw ESCAPES the failover loop:
// submit() rejects, no 'done' event ever reaches the sink, the UI turn
// hangs. With the fix, the 429 falls into the failover path: warn to
// custom2, one 'done', answer = NOTHINK-FAILOVER-OK.

const PORT = parseInt(process.argv[process.argv.length - 1] || "18831", 10);
const BASE = "http://127.0.0.1:" + PORT + "/v1/chat/completions";

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

const server = sofuu.createServer(function (req, res) {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  let last = "";
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (String(msgs[i].role) === "user") { last = String(msgs[i].content || ""); break; }
  }
  /* NOTE: the active provider's MODEL is "o3" (a registry reasoning
   * family). ai.rs is capability-aware on the wire: an effort param for
   * a model UNKNOWN to the registry is stripped before it ever reaches
   * an endpoint (ai.rs openai_effort_and_output_param_wiring), so an
   * invented model name could never reproduce the 400 the noThink
   * branch reacts to. "o3" carries reasoning_effort on the wire. */
  const model = String(parsed.model || "");
  /* The noThink retry omits the reasoning parameter — that absence is
   * the wire-true discriminator between ask 1 and the retry (message
   * counts are not: recall/history injection adds user-role messages). */
  const hasEffort = parsed.reasoning_effort !== undefined;

  if (last.indexOf("MODENOTHINK") >= 0 && model === "o3") {
    if (!hasEffort) {
      console.log("MOCK-NOTHINK: active served 429 on the noThink retry");
      res.writeHead(429, { "Content-Type": "application/json" });
      res.send(JSON.stringify({ error: { message: "mock-active is rate-limited", code: 429 } }));
      return;
    }
    console.log("MOCK-NOTHINK: active served 400 reasoning_effort");
    res.writeHead(400, { "Content-Type": "application/json" });
    res.send(JSON.stringify({ error: { message: "reasoning_effort is not supported by this model", code: 400 } }));
    return;
  }
  console.log("MOCK-NOTHINK: served plain ok (model=" + model + ")");
  sse(res, [
    JSON.stringify({ choices: [{ delta: { content: "NOTHINK-FAILOVER-OK" } }] }),
    JSON.stringify({ usage: { prompt_tokens: 12, completion_tokens: 6 } }),
    "[DONE]",
  ]);
});

async function main() {
  /* Single bind: the endpoint in the harness config points at exactly
   * this port, so a retry on PORT+1 would silently desync the test. */
  server.listen(PORT, "127.0.0.1");
  console.log("MOCK-NOTHINK-READY " + PORT);
}

main().then(async () => {
  const init = sofuu.chat.init({
    host: 'nothink', project: process.env.SOFUU_PROJECT || '/tmp/chat-nothink-e2e',
    permissions: 'prompt',
  });
  if (!init.ok) { console.error("FAIL harness: chat.init failed " + JSON.stringify(init)); process.exit(1); }

  const events = [];
  let submitErr = null;
  let submitRes = null;
  try {
    submitRes = await sofuu.chat.submit("MODENOTHINK hello", {
      noTools: true,
      onEvent: function (e) { events.push(e); },
    });
  } catch (e) { submitErr = e; }

  const warns = events.filter(e => e.kind === "warn").map(e => String((e.payload || {}).message || ""));
  const dones = events.filter(e => e.kind === "done");
  console.log("warns: " + JSON.stringify(warns));
  if (submitErr) console.log("submit threw: " + String((submitErr && submitErr.message) || submitErr));

  check("submit resolved (turn terminated)", !submitErr && !!submitRes && submitRes.ok === true);
  check("noThink warn emitted", warns.some(w => w.indexOf("retrying without effort") >= 0));
  check("retry's 429 fell into the failover loop (warn to custom2)",
        warns.some(w => w.indexOf("failing over to custom2") >= 0));
  check("exactly one done event", dones.length === 1);
  check("done.answer = NOTHINK-FAILOVER-OK",
        dones.length === 1 && dones[0].payload.answer === "NOTHINK-FAILOVER-OK");
  check("done.servedBy = mock-failover",
        dones.length === 1 && dones[0].payload.servedBy === "mock-failover");

  console.log(bad === 0 ? "\nNOTHINK-FAILOVER DRIVER: ALL PASSED"
                        : "\nNOTHINK-FAILOVER DRIVER: " + bad + " FAILED");
  process.exit(bad === 0 ? 0 : 1);
}).catch(e => { console.error("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
