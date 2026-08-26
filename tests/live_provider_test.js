// tests/live_provider_test.js — live-provider E2E for the agent loop.
//
// ENV-GATED: runs ONLY when SOFUU_LIVE_TEST=1 AND a provider API key is
// present in the environment. Otherwise it prints SKIP lines and exits 0,
// so `make test` stays green on machines without credentials / network.
//
//   SOFUU_LIVE_TEST=1 SOFUU_LIVE_PROVIDER=openai SOFUU_LIVE_MODEL=gpt-4o-mini \
//   OPENAI_API_KEY=sk-... ./sofuu run tests/live_provider_test.js
//
// What it verifies against the REAL provider:
//   1. ai.complete round-trip (non-empty text, usage tokens > 0)
//   2. agent loop with an inline tool: plan → tool call → final answer
//      (answer non-empty, usage > 0, trace has plan+tool events, no stop)

const results = { passed: 0, failed: 0, skipped: 0 };
function assert(label, cond) {
  if (cond) { console.log("PASS " + label); results.passed++; }
  else { console.error("FAIL " + label); results.failed++; }
}
function skipSuite(why) {
  console.log("SKIP live_provider_test — " + why);
  console.log("\nLIVE PROVIDER TEST: SKIPPED");
  process.exit(0);
}

const PROVIDER = process.env.SOFUU_LIVE_PROVIDER || "";
const KEYS = [
  ["openai", "OPENAI_API_KEY", "gpt-4o-mini"],
  ["anthropic", "ANTHROPIC_API_KEY", "claude-3-5-haiku-latest"],
  ["openrouter", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"],
];

let provider = PROVIDER;
let apiKey = "";
let defaultModel = "";
if (!provider) {
  for (const [p, envName, model] of KEYS) {
    if (process.env[envName]) { provider = p; apiKey = process.env[envName]; defaultModel = model; break; }
  }
} else {
  const hit = KEYS.find(k => k[0] === provider);
  defaultModel = hit ? hit[2] : "";
  apiKey = process.env[(hit || ["", ""])[1]] || process.env.SOFUU_LIVE_KEY || "";
}
const MODEL = process.env.SOFUU_LIVE_MODEL || defaultModel;

if (process.env.SOFUU_LIVE_TEST !== "1") skipSuite("SOFUU_LIVE_TEST != 1 (opt-in)");
if (!provider) skipSuite("no provider key found (OPENAI/ANTHROPIC/GEMINI/OPENROUTER)");
if (!MODEL) skipSuite("no model resolved — set SOFUU_LIVE_MODEL");

console.log("=== Live Provider Test ===");
console.log("  provider=" + provider + " model=" + MODEL + "\n");

async function main() {
  // ── 1. plain completion round-trip ─────────────────────────────
  {
    const r = await sofuu.ai.complete("Reply with exactly: LIVE-OK", {
      provider, model: MODEL, api_key: apiKey || undefined,
    });
    assert("complete returns text", typeof r.text === "string" && r.text.length > 0);
    assert("complete usage tokens > 0",
      (r.tokens && (r.tokens.input > 0 || r.tokens.output > 0)) ||
      ((r.usage && ((r.usage.promptTokens || 0) + (r.usage.completionTokens || 0)) > 0)));
  }

  // ── 2. agent loop with an inline tool ──────────────────────────
  sofuu.agent.define({
    name: "live-probe",
    system: "You must use the provided tool once, then answer briefly.",
    provider, model: MODEL,
    api_key: apiKey || undefined,
    memory: "off",
    rlm: "off",
    tools: [{
      name: "live_dice",
      description: "Rolls a die. Returns a number 1-6 as JSON.",
      parameters: { type: "object", properties: {}, required: [] },
      execute: async () => JSON.stringify({ roll: 1 + Math.floor(Math.random() * 6) }),
    }],
    budget: { maxSteps: 4, maxDepth: 1, maxTokens: 200000, maxWallMs: 120000 },
  });
  const res = await sofuu.agent.run("live-probe",
    "Call live_dice, then reply with exactly: LIVE-AGENT-OK <roll>");
  assert("agent answer non-empty", typeof res.answer === "string" && res.answer.length > 0);
  assert("agent used the tool", (res.usage && res.usage.toolCalls > 0) ||
    (res.trace || []).some(e => e.kind === "tool"));
  assert("agent made LLM calls", (res.usage && res.usage.llmCalls > 0));
  assert("agent not stopped by budget/cancel", !res.stopped);
  const kinds = (res.trace || []).map(e => e.kind);
  assert("trace has plan events", kinds.indexOf("plan") >= 0);

  console.log("\n=== RESULTS ===");
  console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
  if (results.failed > 0) process.exit(1);
  console.log("\nLIVE PROVIDER TEST: ALL PASSED (" + provider + "/" + MODEL + ")");
}

main().catch(err => {
  console.error("FAIL exception: " + (err && err.message ? err.message : err));
  process.exit(1);
});
