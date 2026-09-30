// tests/caps_test.js — the context-window evidence ladder through the
// REAL native seam (sofuu.ai.resolveCaps + sofuu.ml.alloc.ingestListing).
// This is the regression net for the reported bug: a 1M model reported as
// 32k, and `/ctx 1000000` appearing to do nothing.
//
//   1. explicit ctx is obeyed even above the strongest evidence
//   2. an INHERITED global still shrinks to the model's real bound
//   3. endpoint-published caps are detected and adopted (the F2 harvest)
//   4. an unknown model gets the honest conservative default
//   5. configExceedsEvidence carries the bound for the advisory line
//
// Run:  ./sofuu run tests/caps_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== Context-window ladder Test ===\n");

assert("sofuu.ai.resolveCaps is function", typeof sofuu.ai.resolveCaps === "function");
assert("sofuu.ml.alloc.ingestListing is function",
    typeof sofuu.ml.alloc.ingestListing === "function");

const caps = (model, url, win, out, explicit) =>
    JSON.parse(sofuu.ai.resolveCaps(model || "", url || "", win || 0, out || 0, !!explicit));

// Synthetic ids — no registry entry exists for either, so only what the
// test itself feeds the store can be evidence.
const BIG = "caps-test-vendor/caps-test-million-zz1";
const SMALL = "caps-test-vendor/caps-test-small-zz2";
const ROOT = "https://caps-test.example.com/v1";
const ROOT2 = "https://caps-test-two.example.com/v1";

// ── 1 + 5: explicit raise is obeyed, with the evidence reported ──
let r = caps(BIG, ROOT, 1048576, 0, true);
assert("explicit /ctx 1M is obeyed on an unknown model", r.window === 1048576);
assert("explicit value keeps winSource=config", r.winSource === "config");

// The endpoint publishes real caps for the same model…
let n = sofuu.ml.alloc.ingestListing(ROOT, JSON.stringify({
    data: [{ id: BIG, context_length: 1048576, max_completion_tokens: 65536 }],
}));
assert("harvest ingested the listing entry", n >= 1);

// …so the evidence now matches the request: no advisory needed.
r = caps(BIG, ROOT, 1048576, 0, true);
assert("detected endpoint caps = 1M", r.window === 1048576);
assert("no advisory when explicit value matches evidence", r.clampedConfig === false);

// A DIFFERENT endpoint publishes a small window for the same id: the
// inherited global must shrink, the explicit value must not.
n = sofuu.ml.alloc.ingestListing(ROOT2, JSON.stringify({
    data: [{ id: BIG, context_length: 32768 }],
}));
assert("second root ingested", n >= 1);
r = caps(BIG, ROOT2, 1048576, 0, true);
assert("explicit value still honored above a smaller endpoint's caps", r.window === 1048576);
assert("advisory fires: clampedConfig true", r.clampedConfig === true);
assert("advisory carries the evidence bound (32768)", r.configExceedsEvidence === 32768);
r = caps(BIG, ROOT2, 1048576, 0, false);
assert("inherited global shrinks to the real bound", r.window === 32768);
assert("shrunk value reports the evidence source", r.winSource === "discovered");

// ── 2: the ring-bug guard, unchanged ──
// Unknown model, nothing known anywhere: an inherited global is the ONLY
// input, so it stands — and it is reported as coming from config, not
// from detection.
r = caps(SMALL, ROOT, 1048576, 0, false);
assert("no evidence anywhere: the configured value stands", r.window === 1048576);
assert("…and is attributed to config, not to detected truth",
    r.winSource === "config" && r.source === "config");
assert("…with no advisory (nothing contradicts it)", r.clampedConfig === false);
assert("output side stays unknown → max_source default",
    r.maxSource === "default" && r.maxOutput === 4096);
// The real ring bug needs CONTRADICTING evidence: this endpoint publishes
// 32k for that model, so the stale 1M global must shrink to it.
n = sofuu.ml.alloc.ingestListing(ROOT, JSON.stringify({
    data: [{ id: SMALL, context_length: 32768, max_tokens: 4096 }],
}));
assert("small-model caps ingested", n >= 1);
r = caps(SMALL, ROOT, 1048576, 0, false);
assert("inherited 1M global never shadows a 32k model", r.window === 32768);
assert("shrunk to the endpoint's published window", r.winSource === "discovered");
assert("with no config at all it is the honest 32k default",
    caps(SMALL, ROOT, 0, 0, false).window === 32768);

// ── 3: a plain detection (no config) adopts what the endpoint says ──
r = caps(BIG, ROOT, 0, 0, false);
assert("no config + endpoint caps = 1M detected", r.window === 1048576);
assert("detected window is marked known", r.known === true);
assert("max output comes from the listing too", r.maxOutput === 65536);

// ── 4: explicit below the ceiling is honored quietly ──
r = caps(BIG, ROOT, 32768, 0, true);
assert("explicit value below the ceiling honored", r.window === 32768);
assert("budgeting under the ceiling raises no advisory", r.clampedConfig === false);

// ── explicit output cap above evidence ──
r = caps(BIG, ROOT, 0, 200000, true);
assert("explicit /maxout above evidence is honored", r.maxOutput === 200000);
r = caps(BIG, ROOT, 0, 200000, false);
assert("inherited /maxout shrinks to the listing's cap", r.maxOutput === 65536);

console.log("\n=== RESULTS ===");
console.log("Passed: " + results.passed + " | Failed: " + results.failed);
if (results.failed > 0) process.exit(1);
console.log("\n✅ All context-window ladder tests PASSED");
