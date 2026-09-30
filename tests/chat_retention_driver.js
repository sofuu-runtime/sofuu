// tests/chat_retention_driver.js — driver for the Claude-style
// transcript-retention E2E (tests/chat_retention_e2e.sh).
//
// Layer 1 (agent): the shipped agent.js returns res.transcript — the
//   turn's assistant tool_calls + tool messages (plain copies).
// Layer 2 (chat driver): after a tool turn, the transcript PERSISTS into
//   history — the next turn's outbound request must still carry the
//   tool messages (user → assistant+tc → tool → assistant ordering,
//   no orphaned tool message). The mock answers RETENTION-SAW-NEEDLE
//   only when the needle (a tool result) is present in the request.
const PORT = process.argv[process.argv.length - 1];
const BASE = "http://127.0.0.1:" + PORT + "/v1/chat/completions";
const PROJ = process.env.RETENTION_PROJ || "";
let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

async function main() {
  /* ── Layer 1: agent.run returns the turn transcript ─────────── */
  sofuu.agent.define({
    name: "tt-agent",
    system: "AGENT=tt You read files.",
    tools: [{
      name: "read_file",
      description: "Read a file",
      parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
      execute: async (args) => "NEEDLE-TT-88 notes content for path " + (args && args.path),
    }],
    memory: "off", rlm: "off",
    provider: "openai", model: "mock", api_key: "x", base_url: BASE,
  });
  const a1 = await sofuu.agent.run("tt-agent", "USE-TOOL-TT please", {});
  check("L1 turn answered via the tool result",
        String(a1.answer || "").indexOf("RETENTION-SAW-NEEDLE") >= 0);
  check("L1 transcript is an array of 2", Array.isArray(a1.transcript) && a1.transcript.length === 2);
  check("L1 transcript[0] assistant w/ tool_calls",
        a1.transcript && a1.transcript[0].role === "assistant" &&
        Array.isArray(a1.transcript[0].tool_calls) && a1.transcript[0].tool_calls.length === 1);
  check("L1 transcript[1] tool msg with matching id",
        a1.transcript && a1.transcript[1].role === "tool" &&
        a1.transcript[1].tool_call_id === a1.transcript[0].tool_calls[0].id);
  check("L1 transcript tool result carries the needle",
        a1.transcript && String(a1.transcript[1].content || "").indexOf("NEEDLE-TT-88") >= 0);
  if (Array.isArray(a1.transcript)) a1.transcript.length = 0; /* snapshot check */
  const a2 = await sofuu.agent.run("tt-agent", "USE-TOOL-TT again", {});
  check("L1 transcript snapshot is per-run",
        Array.isArray(a2.transcript) && a2.transcript.length === 2);

  /* ── Layer 2: the chat driver persists transcripts ─────────── */
  await sofuu.chat.init({ host: "e2e", project: PROJ });
  sofuu.chat.setPermissions("full");
  const t1 = await sofuu.chat.submit("USE-TOOL-TT read notes", { onEvent: () => {} });
  check("L2 tool turn 1 answered", String(t1.answer || "").indexOf("RETENTION-SAW-NEEDLE") >= 0);
  const st = sofuu.chat.state({ history: true });
  const hist = st.history || [];
  check("L2 history has the tool block (>= 4 entries)",
        hist.length >= 4);
  check("L2 history user → assistant+tc → tool → assistant ordering",
        hist.length >= 4 && hist[hist.length - 4].role === "user" &&
        hist[hist.length - 3].role === "assistant" && Array.isArray(hist[hist.length - 3].tool_calls) &&
        hist[hist.length - 2].role === "tool" && hist[hist.length - 1].role === "assistant");
  check("L2 history tool result carries the needle",
        hist.some(m => m.role === "tool" && String(m.content || "").indexOf("NEEDLE-TT-88") >= 0));
  let orphan = false;
  let prevTC = false;
  for (const m of hist) {
    if (m.role === "tool" && !prevTC) { orphan = true; break; }
    prevTC = !!(m.tool_calls && m.tool_calls.length);
  }
  check("L2 history has no orphaned tool message", !orphan);

  /* The proof turn: a PLAIN follow-up. The mock answers RETENTION-SAW-
   * NEEDLE only when the request contains the needle — with retention
   * the plain turn's request carries turn 1's tool result. */
  const t2 = await sofuu.chat.submit("now just summarize it", { onEvent: () => {} });
  check("L2 plain turn 2 request carried the retained transcript",
        String(t2.answer || "").indexOf("RETENTION-SAW-NEEDLE") >= 0);

  /* Compaction safety: /compact folds whole blocks — after compaction
   * the history must still be orphan-free (the summary is a system
   * entry; the newest block stays). */
  const cp = await sofuu.chat.compact();
  console.log("  compact() -> " + JSON.stringify(cp));
  const hist2 = ((sofuu.chat.state({ history: true }) || {}).history) || [];
  let orphan2 = false;
  prevTC = false;
  for (const m of hist2) {
    if (m.role === "tool" && !prevTC) { orphan2 = true; break; }
    prevTC = !!(m.tool_calls && m.tool_calls.length);
  }
  check("L2 post-compact history still orphan-free", !orphan2);
  check("L2 compact ok or nothing to fold",
        !!(cp && (cp.ok || /nothing to compact/.test(String(cp.error || "")))));

  const t3 = await sofuu.chat.submit("and once more", { onEvent: () => {} });
  check("L2 turn 3 answered after compaction",
        String(t3.answer || "").length > 0);

  console.log("\nRETENTION E2E: " + (bad ? "FAILED" : "PASS") + " (" + ok + " checks)");
  process.exit(bad ? 1 : 0);
}
main().catch(e => { console.error("ERROR", e && (e.stack || e.message)); process.exit(1); });
