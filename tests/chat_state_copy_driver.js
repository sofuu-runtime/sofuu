// tests/chat_state_copy_driver.js — js-4 (AUDIT-2026-09-07) RED/GREEN
// driver, run by tests/chat_state_copy_e2e.sh (which provides HOME +
// config + cwd = project). Drives the JS chat engine (sofuu.chat.submit —
// the desktop driver), NOT the TUI's Rust driver.
//
// js-4: sofuu.chat.state({history:true}) documents "Copies, never live
// references", but the per-message map passed m.tool_calls through BY
// REFERENCE. A host mutating the returned array would silently rewrite
// the live S.history entries that the NEXT turn sends to the provider.
//
// Scenario (one server, branched on REQUEST CONTENT — the config uses a
// single provider for both turns, so model-name branching can't work):
//   turn 1, request 1 (messages hold no assistant tool_calls): serve a
//     write_file tool_call writing STATE-CANARY-ORIGINAL into canary.txt;
//     permissionProfile 'full' auto-approves it. The engine's post-tool
//     round trip still carries the tool_calls in its transcript but only
//     ONE user message — answered plainly so the turn settles.
//   After submit 1 the driver reads state({history:true}), finds the
//   tool_calls entry, and MUTATES the returned copy in place (rewrites
//   the function name to read_file and the arguments JSON so the second
//   turn's request would contain the corrupted pair if state() leaks).
//   turn 2 (TWO user messages in the transcript): the server asserts —
//   server-side, Mimosa-XSS-safe — that the retained history still
//   carries the ORIGINAL tool_call name/arguments, replying with one of
//   two constant sentences the driver checks.
//
// With the leak, turn 2 sees name=read_file + forged args →
// "STATE-CANARY-CORRUPTED". With the fix (deep copy), turn 2 sees the
// original pair → "STATE-CANARY-INTACT".

const PORT = parseInt(process.argv[process.argv.length - 1] || "18971", 10);
const SENTINEL = "STATE-CANARY-INTACT";
const CORRUPT = "STATE-CANARY-CORRUPTED";

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

let sawCorruption = false;
let toolServed = 0;
const server = sofuu.createServer(function (req, res) {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];
  const hasTc = msgs.some(m => m.role === "assistant" && m.tool_calls && m.tool_calls.length);
  /* The engine injects its own user-role messages (recalls, footers), so
   * counting user messages can't discriminate the turns. The driver puts
   * a marker in the second submit's text instead — request content, not
   * model name (one config provider serves both turns). */
  const isTurn2 = msgs.some(m => m.role === "user" &&
    String(m.content || "").indexOf("second turn") >= 0);

  if (!isTurn2) {
    /* Turn 1: first request gets the tool call; the post-tool round trip
     * (transcript already has assistant tool_calls) gets a plain answer
     * so the turn settles. */
    if (!hasTc && toolServed === 0) {
      toolServed++;
      const tcs = [{
        index: 0, id: "call_state_1", type: "function",
        function: {
          name: "write_file",
          arguments: JSON.stringify({ path: "canary.txt", content: "STATE-CANARY-ORIGINAL" }),
        },
      }];
      sse(res, [
        JSON.stringify({ choices: [{ delta: { tool_calls: tcs } }] }),
        JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 8 } }),
        "[DONE]",
      ]);
      return;
    }
    sse(res, [
      JSON.stringify({ choices: [{ delta: { content: "TURN1-SETTLED" } }] }),
      JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } }),
      "[DONE]",
    ]);
    return;
  }

  /* Turn 2: assert server-side on the retained history's tool_calls. */
  let tcName = null, tcArgs = null;
  for (const m of msgs) {
    if (m.role === "assistant" && m.tool_calls && m.tool_calls.length) {
      tcName = String(m.tool_calls[0].function.name);
      tcArgs = String(m.tool_calls[0].function.arguments);
    }
  }
  console.log("MOCK-STATE: turn2 saw tool_call name=" + tcName +
              " args=" + JSON.stringify(tcArgs));
  if (tcName !== "write_file" || tcArgs.indexOf("STATE-CANARY-ORIGINAL") < 0) {
    sawCorruption = true;
  }
  sse(res, [
    JSON.stringify({ choices: [{ delta: { content: sawCorruption ? CORRUPT : SENTINEL } }] }),
    JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 4 } }),
    "[DONE]",
  ]);
});

async function main() {
  server.listen(PORT, "127.0.0.1");
  console.log("MOCK-STATE-READY " + PORT);

  const proj = process.env.SOFUU_PROJECT || process.cwd();

  const init = sofuu.chat.init({
    host: 'st', project: proj,
    /* full: tools auto-pass (no approvals needed for this driver). */
    permissionProfile: 'full',
  });
  if (!init.ok) { console.error("FAIL harness: chat.init failed " + JSON.stringify(init)); process.exit(1); }

  /* Turn 1: real tool round → S.history gains the assistant tool_calls. */
  const r1 = await sofuu.chat.submit("STATE-CANARY make the file", {});
  check("turn 1 done", !!(r1 && r1.ok === true));

  /* Snapshot the state copy and vandalize it the way a careless host
   * would: point the returned tool_calls at a forged call. */
  const st = sofuu.chat.state({ history: true });
  let found = false;
  if (st.history) {
    for (const m of st.history) {
      if (m.role === "assistant" && m.tool_calls && m.tool_calls.length) {
        found = true;
        m.tool_calls[0].function.name = "read_file";
        m.tool_calls[0].function.arguments =
          JSON.stringify({ path: "canary.txt" });
        m.content = "VANDALIZED";
      }
    }
  }
  check("state({history:true}) exposed a tool_calls entry", found);

  /* Turn 2: plain answer; the server asserts on what the ENGINE still
   * holds (not on the vandalized copy). */
  const r2 = await sofuu.chat.submit("STATE-CANARY second turn", {});
  check("turn 2 done", !!(r2 && r2.ok === true));
  check("turn 2 answer reports intact engine history",
        !!(r2 && r2.ok === true && r2.answer === SENTINEL));
  check("engine history survived the state() mutation (no corruption seen by mock)",
        sawCorruption === false);

  console.log(bad === 0 ? "\nSTATE-COPY DRIVER: ALL PASSED"
                        : "\nSTATE-COPY DRIVER: " + bad + " FAILED");
  process.exit(bad === 0 ? 0 : 1);
}

main().catch(e => { console.error("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
