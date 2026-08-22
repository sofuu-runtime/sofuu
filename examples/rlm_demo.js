// examples/rlm_demo.js — RLM needle-in-a-haystack demo (PLAN-RLM R2).
//
// Builds a ~400KB synthetic report, plants one code word at ~70%, and asks
// sofuu.rlm.query to find it. The context never enters a prompt window
// whole — the model reads it through the RLM sandbox. No chat, no TUI:
// this is the headless entry point.
//
// Provider comes from the environment — there is deliberately NO default
// (no provider/model is built in or implied anywhere in the runtime):
//   SOFUU_PROVIDER   (required, e.g. "openai", "ollama", "custom")
//   SOFUU_MODEL      (required, e.g. "gpt-4o", "llama3")
//   SOFUU_BASE_URL   (optional endpoint override)
//   SOFUU_API_KEY    (optional; env vars such as OPENAI_API_KEY also work)
//
// Run:  SOFUU_PROVIDER=... SOFUU_MODEL=... ./sofuu run examples/rlm_demo.js

const provider = process.env.SOFUU_PROVIDER || "";
const model = process.env.SOFUU_MODEL || "";
const NEEDLE = "The code word is ULTRAVIOLET-PIKE-417.";

/* ~400KB: 380 sections of ~1.1KB of filler, needle planted at ~70%. */
function buildContext() {
  const parts = [];
  for (let i = 1; i <= 380; i++) {
    parts.push(
      "Section " + i + " — quarterly operations report\n" +
      ("Filler paragraph for section " + i + ": routine metrics, staffing notes, " +
       "facility updates and other unremarkable details. ").repeat(6)
    );
    if (i === 266) parts.push(NEEDLE);   /* ~70% of the way through */
  }
  return parts.join("\n\n");
}

async function main() {
  if (!provider || !model) {
    console.error("Set SOFUU_PROVIDER and SOFUU_MODEL (e.g. SOFUU_PROVIDER=openai SOFUU_MODEL=gpt-4o) — the runtime ships no default model.");
    process.exit(2);
  }
  const context = buildContext();
  const question = "What is the code word mentioned in this report? Answer with just the code word.";
  console.log("Context: " + context.length + " chars (" + Math.round(sofuu.ai.estimateTokens(context)) + " est. tokens)");
  console.log("Provider: " + provider + " · model: " + model + "\n");

  const opts = { provider: provider, model: model, trace: true };
  if (process.env.SOFUU_BASE_URL) opts.base_url = process.env.SOFUU_BASE_URL;
  if (process.env.SOFUU_API_KEY) opts.api_key = process.env.SOFUU_API_KEY;

  const res = await sofuu.rlm.query(context, question, opts);
  console.log("Answer: " + res.answer);
  console.log("\n{ calls: " + res.calls + ", rounds: " + res.rounds + ", ms: " + res.ms +
              (res.stopped ? ", stopped: " + res.stopped : "") + " }");
  if (res.trace) {
    console.log("Trace (" + res.trace.length + " events):");
    for (const e of res.trace.slice(0, 12))
      console.log("  t=" + e.t + "ms  " + e.kind + "  " + String(e.repr).slice(0, 100));
    if (res.trace.length > 12) console.log("  … " + (res.trace.length - 12) + " more");
  }
}

main().catch(e => { console.error("rlm_demo failed: " + (e && e.stack || e)); process.exit(1); });
