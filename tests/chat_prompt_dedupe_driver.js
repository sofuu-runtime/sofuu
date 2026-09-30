// tests/chat_prompt_dedupe_driver.js — driver for the prompt-dedupe E2E
// (session-2, AUDIT-2026-09-07). Drives the shipped chat.js against
// tests/mock_prompt_dedupe_server.js in two modes (DEDUPE_MODE env):
//
//   think     — config model "o3" + effort high: the first wire request
//               carries reasoning_effort, the mock rejects it with HTTP 400
//               ("does not support reasoning_effort" → THINK class), the
//               driver retries WITHOUT effort and gets PLAIN-OK. The
//               duplicate-prompt bug would log TWO prompt events here →
//               resume().turns would read 2 instead of 1.
//   capacity  — config provider p1/model mock-p1 with failover entry
//               p2/mock-p2: mock-p1 answers HTTP 401 (CAPACITY class) →
//               failover → PLAIN-OK on mock-p2. Two plain submits then
//               prove the gate doesn't over-suppress (turns 1 → 2).
//
// Run: ./sofuu run tests/chat_prompt_dedupe_driver.js <port>

const PORT = process.argv[process.argv.length - 1];
const PROJ = process.env.DEDUPE_PROJ || "";
const MODE = process.env.DEDUPE_MODE || "think";
let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

async function main() {
  await sofuu.chat.init({ host: "e2e", project: PROJ });
  sofuu.chat.setPermissions("full");

  if (MODE === "think") {
    const r1 = await sofuu.chat.submit("DEDUPE-THINK-1 please answer plainly", { onEvent: () => {} });
    console.log("r1.answer:", JSON.stringify(String(r1.answer || "").slice(0, 80)));
    check("think turn completed after the no-effort retry (PLAIN-OK)",
          String(r1.answer || "").indexOf("PLAIN-OK") >= 0);
    const sid = sofuu.chat.state().sessionId;
    check("session id present", !!sid);
    const rr = sofuu.chat.resume(sid);
    check("think: resume turns === 1, not 2 (got " + JSON.stringify(rr) + ")",
          !!(rr && rr.ok) && rr.turns === 1);
  } else {
    const r1 = await sofuu.chat.submit("DEDUPE-CAP-1 first turn", { onEvent: () => {} });
    console.log("r1.answer:", JSON.stringify(String(r1.answer || "").slice(0, 80)));
    check("capacity turn 1 failed over to p2 and completed (PLAIN-OK)",
          String(r1.answer || "").indexOf("PLAIN-OK") >= 0);
    const sid = sofuu.chat.state().sessionId;
    check("session id present", !!sid);
    const rr1 = sofuu.chat.resume(sid);
    check("capacity: resume turns === 1 after the failover turn (got " +
          JSON.stringify(rr1) + ")", !!(rr1 && rr1.ok) && rr1.turns === 1);
    const r2 = await sofuu.chat.submit("DEDUPE-CAP-2 second plain turn", { onEvent: () => {} });
    check("capacity turn 2 plain (PLAIN-OK)",
          String(r2.answer || "").indexOf("PLAIN-OK") >= 0);
    const rr2 = sofuu.chat.resume(sid);
    check("capacity: resume turns === 2 after the second submit (got " +
          JSON.stringify(rr2) + ") — gate does not over-suppress",
          !!(rr2 && rr2.ok) && rr2.turns === 2);
  }

  console.log("\nPROMPT-DEDUPE (" + MODE + "): " + (bad ? "FAILED" : "PASS") + " (" + ok + " checks)");
  process.exit(bad ? 1 : 0);
}
main().catch(e => { console.error("ERROR", e && e.message); process.exit(1); });
