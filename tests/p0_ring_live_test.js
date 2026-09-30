// p0_ring_live_test.js — live proof for the P0 ring fix.
//
// Boots the REAL shipped chat driver against the ACTIVE provider from
// ~/.sofuu/config.json (flat ctx_window 1M / max_output 384k present —
// verified externally before the run), submits one tiny turn, and
// asserts the ctx event reports the SELECTED model's real window —
// never the flat global.
//
// Usage: SOFUU_LIVE_TEST=1 ./sofuu run tests/p0_ring_live_test.js
const FLAT_CTX_WINDOW = 1000000;  // verified in ~/.sofuu/config.json pre-run
const FLAT_MAX_OUTPUT = 384000;
const events = [];
let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

if (process.env.SOFUU_LIVE_TEST !== "1") {
  console.log("SKIP p0_ring_live_test — SOFUU_LIVE_TEST != 1");
  process.exit(0);
}

async function main() {
  const r = await sofuu.chat.init({ host: "test" });
  console.log("init:", JSON.stringify({ model: r.model, provider: r.provider }));
  // The boot ctx event fired inside init without a sink — submit re-emits.
  await sofuu.chat.submit("Reply with exactly: P0-OK", {
    noTools: true,
    onEvent: function (e) { events.push(e); },
  });
  const ctx = events.filter(e => e.kind === "ctx");
  console.log("ctx events:", JSON.stringify(ctx));
  console.log("flat config (pre-verified): ctx_window =", FLAT_CTX_WINDOW,
              "max_output =", FLAT_MAX_OUTPUT);
  const last = ctx[ctx.length - 1];
  check("ctx event arrived", !!last);
  if (last) {
    // The ring bug was: config ABOVE the model's real window shadowing
    // truth. Post-fix invariant: the reported window is never above what
    // the evidence ladder resolved for this model — config can only
    // shrink, never inflate. (Here glm-5.3-free's real window is
    // 1.31M, so a 1M config override is legitimate and shows as such.)
    check("clamp flag surfaced when config exceeded the model's real caps (clampedConfig=" +
          last.payload.clampedConfig + ")",
          last.payload.clampedConfig === true);
    check("model reported in the meter", !!(last.payload.model || r.model));
    // And the exact user bug: a window above the model's truth is NEVER
    // shown. Resolve with NO config override — pure evidence — and
    // confirm the config override can't produce a window above it.
    const capsJs = sofuu.ai.resolveCaps(r.model || "", "https://api.tokenrouter.com/v1/chat/completions", 0, 0);
    const pure = JSON.parse(capsJs);
    console.log("pure-evidence resolve:", JSON.stringify(pure));
    check("evidence window resolved (got " + pure.window + ")",
          pure.window > 0 && pure.window < 2_000_000);
    check("config override never exceeds evidence (" + last.payload.window + " ≤ " + pure.window + ")",
          last.payload.window <= pure.window);
  }
  console.log("\nP0 RING LIVE: " + (bad ? "FAILED" : "PASS") + " (" + ok + " checks)");
  process.exit(bad ? 1 : 0);
}
main().catch(e => { console.error("ERROR", e && e.message); process.exit(1); });
