// failover_live_test.js — live proof of the provider failover chain.
//
// Precondition (set by the operator, not this file): the ACTIVE provider
// entry's api_key has been replaced with a broken value; every other
// configured provider still carries its real key. One turn must:
//   1. die on the active provider (auth error)
//   2. emit a warn announcing the failover to the next provider
//   3. complete with a 'done' whose servedBy is NOT the broken model
const events = [];
let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}
if (process.env.SOFUU_LIVE_TEST !== "1") {
  console.log("SKIP failover_live_test — SOFUU_LIVE_TEST != 1");
  process.exit(0);
}
async function main() {
  const r = await sofuu.chat.init({ host: "test" });
  console.log("active:", r.provider, "/", r.model);
  await sofuu.chat.submit("Reply with exactly: FAILOVER-OK", {
    noTools: true,
    onEvent: function (e) { events.push(e); },
  });
  const warns = events.filter(e => e.kind === "warn").map(e => e.payload.message);
  console.log("warns:", JSON.stringify(warns));
  const dones = events.filter(e => e.kind === "done");
  check("turn completed", dones.length > 0);
  check("failover warn emitted", warns.some(w => /failing over to/i.test(w || "")));
  if (dones.length) {
    const p = dones[dones.length - 1].payload;
    console.log("done payload keys:", Object.keys(p).join(","));
    check("servedBy present (" + (p.servedBy || "?") + ")", !!p.servedBy);
    check("servedBy is NOT the broken active model (" + r.model + ")",
          p.servedBy !== r.model);
    check("answer delivered", (p.answer || "").length > 0);
  }
  console.log("\nFAILOVER LIVE: " + (bad ? "FAILED" : "PASS") + " (" + ok + " checks)");
  process.exit(bad ? 1 : 0);
}
main().catch(e => { console.error("ERROR", e && e.message); process.exit(1); });
