// tests/chat_resume_driver.js — driver for the structured-resume E2E.
// Drives the shipped chat.js against tests/mock_resume_server.js:
//   turn 1: tool turn (read_file notes.txt, canary content)
//   turn 2: plain turn
//   resume: reload the session from its .qtsq
//   turn 3: the resume context block (tool activity + canary) must reach
//           the model — the mock answers RESUME-CONTEXT-OK only then.
const PORT = process.argv[process.argv.length - 1];
const BASE = "http://127.0.0.1:" + PORT + "/v1/chat/completions";
const PROJ = process.env.RESUME_PROJ || "";
let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

async function main() {
  await sofuu.chat.init({ host: "e2e", project: PROJ });
  sofuu.chat.setPermissions("full");
  const r1 = await sofuu.chat.submit("USE-TOOL-PLEASE", { onEvent: () => {} });
  console.log("r1.answer:", JSON.stringify(String(r1.answer || "").slice(0, 120)));
  check("turn 1 completed the tool round (TOOL-RAN-OK)",
        String(r1.answer || "").indexOf("TOOL-RAN-OK") >= 0);
  const r2 = await sofuu.chat.submit("and now just say something", { onEvent: () => {} });
  console.log("r2.answer:", JSON.stringify(String(r2.answer || "").slice(0, 120)));
  check("turn 2 plain (PLAIN-OK)", String(r2.answer || "").indexOf("PLAIN-OK") >= 0);

  const sid = sofuu.chat.state().sessionId;
  check("session id present", !!sid);
  /* Debug: replay the exact save→load the driver does. */
  try {
    const st = sofuu.chat.state();
    const proj = st.project;
    const fpath = proj + "/.sofuu/sessions/" + sid + ".qtsq";
    let h = 0x811c9dc5n; // placeholder — real fnv1a64 lives in chat.js
    console.log("debug: file=" + fpath + " len=" + fpath.length);
    const exists = __session_read_file ? !!__session_read_file(proj + "/.sofuu/sessions/registry.json") : "?";
    console.log("debug: registry readable via __session_read_file: " + exists);
  } catch (eD) { console.log("debug probe error: " + (eD && eD.message)); }
  const rr = sofuu.chat.resume(sid);
  check("resume ok (" + JSON.stringify(rr) + ")", !!(rr && rr.ok) && rr.turns === 2);

  const r3 = await sofuu.chat.submit("what did you do before?", { onEvent: () => {} });
  check("turn 3: tool activity + canary reached the model (RESUME-CONTEXT-OK)",
        String(r3.answer || "").indexOf("RESUME-CONTEXT-OK") >= 0);

  console.log("\nRESUME E2E: " + (bad ? "FAILED" : "PASS") + " (" + ok + " checks)");
  process.exit(bad ? 1 : 0);
}
main().catch(e => { console.error("ERROR", e && e.message); process.exit(1); });
