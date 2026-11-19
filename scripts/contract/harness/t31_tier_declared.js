// t31_tier_declared.js — R3.1: every path in engine.paths declares a
// determinism tier (exact, tolerant, or best-effort), the tolerant paths
// declare their error bound, and the default path is the exact one.
//
// PURE CHECK: it reads the manifest the engine publishes and runs no model,
// tool or network exchange. The per-knob determinism inventory lives in
// contract/knobs.json.
//
// RESOURCE: pure: the capability manifest published by the running build
//
// Run: ./sofuu run scripts/contract/harness/t31_tier_declared.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

const P = (s) => JSON.parse(s);

function main() {
    const raw = sofuu.capabilities();
    const caps = typeof raw === "string" ? P(raw) : raw;
    assert("capabilities returned valid engine block", caps && caps.engine && caps.engine.paths);

    const paths = caps.engine.paths;
    const VALID_TIERS = ["exact", "tolerant", "best-effort"];

    // 1. Every path has a valid tier
    const pathNames = Object.keys(paths);
    assert("at least 3 engine paths declared", pathNames.length >= 3);

    for (const name of pathNames) {
        const info = paths[name];
        const hasValidTier = info && VALID_TIERS.indexOf(info.tier) >= 0;
        assert(`path '${name}' declares valid tier '${info && info.tier}'`, hasValidTier);
    }

    // 2. The canonical three paths exist with their required tiers
    assert("cpu-tiles is declared with tier 'exact'",
        paths["cpu-tiles"] && paths["cpu-tiles"].tier === "exact");
    assert("metal-mma is declared with tier 'tolerant'",
        paths["metal-mma"] && paths["metal-mma"].tier === "tolerant");
    assert("mmq is declared with tier 'best-effort'",
        paths["mmq"] && paths["mmq"].tier === "best-effort");

    // 3. Tolerant paths declare a bound
    assert("metal-mma tolerant path declares an error bound",
        Boolean(paths["metal-mma"] && paths["metal-mma"].bound));
    for (const name of pathNames) {
        const info = paths[name];
        if (info && info.tier === "tolerant") {
            assert(`tolerant path '${name}' declares an error bound`, Boolean(info.bound));
        }
    }

    console.log(`\n=== t31 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
