// t63_golden.js — R6.7/T6.3: the deterministic trace renderer matches the
// committed golden fixture, and rendering is stable across runs.
//
// Run: ./sofuu run scripts/contract/harness/t63_golden.js
// Fixtures: contract/fixtures/golden-trace.ndjson + golden-trace.txt
// (regenerate the .txt only when the render contract changes — the bytes are
// the contract).
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG"); process.exit(1); }, 15000);

async function main() {
    const ndjson = await sofuu.fs.readFile("contract/fixtures/golden-trace.ndjson");
    const golden = await sofuu.fs.readFile("contract/fixtures/golden-trace.txt");

    const events = String(ndjson).split("\n").filter((l) => l.trim() !== "").map((l) => JSON.parse(l));

    // Render from the committed events — twice, for determinism.
    const r1 = sofuu.events.render(JSON.stringify(events));
    const r2 = sofuu.events.render(JSON.stringify(events));
    assert("render is deterministic", r1 === r2);

    // Seq order + shape: one line per event, stable prefix.
    const lines = String(r1).split("\n").filter((l) => l.trim() !== "");
    assert("one line per event", lines.length === events.length);
    assert("lines are seq-prefixed and ordered",
        lines.every((l, i) => l.indexOf(String(events[i].seq).padStart(4, "0")) === 0));
    assert("kinds render in the fixed column", lines[0].indexOf("delta") === 5);

    // The golden bytes: the render contract is pinned, byte-for-byte.
    assert("render matches the committed golden bytes", String(r1) === String(golden));
    if (String(r1) !== String(golden)) {
        console.error("--- got ---\n" + r1 + "\n--- want ---\n" + golden);
    }

    // Canonical usage: the three producer shapes unify (T6.7).
    const a = JSON.parse(sofuu.events.canonicalUsage(JSON.stringify({ in: 3, out: 4, model: "m", provider: "local" })));
    const b = JSON.parse(sofuu.events.canonicalUsage(JSON.stringify({ pt: 3, ct: 4, model: "m" })));
    const c = JSON.parse(sofuu.events.canonicalUsage(JSON.stringify({ promptTokens: 3, completionTokens: 4 })));
    assert("usage shapes unify (in/out)", a.in === b.in && a.in === c.in && a.out === b.out && a.out === c.out);
    assert("local cost semantics: cost 0", a.cost_usd === 0 && b.cost_usd === 0 && c.cost_usd === 0);
    const cloud = JSON.parse(sofuu.events.canonicalUsage(JSON.stringify({ promptTokens: 5, completionTokens: 7, costUsd: 0.0135 })));
    assert("cloud cost carried", cloud.cost_usd === 0.0135 && cloud.provider === "unknown");

    console.log(`\n=== t63 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main();
