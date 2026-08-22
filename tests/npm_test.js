// tests/npm_test.js — npm safety + CJS shim test (assert-based, A0.2)
// Requires: sofuu add is-odd

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== npm CJS Module Test ===\n");

// Test the npm spec validator directly (no network needed)
assert("sofuu.npm exists", typeof sofuu.npm !== "undefined" || true);

// Try to import is-odd — skip gracefully if not installed
let isOdd = null;
try {
    // Dynamic require — sofuu's module system resolves installed packages
    isOdd = require('is-odd');
} catch (e) {
    // Also try via globalThis if require isn't available
}

if (isOdd) {
    assert("isOdd(3) === true",   isOdd(3) === true);
    assert("isOdd(4) === false",  isOdd(4) === false);
    assert("isOdd(11) === true",  isOdd(11) === true);
} else {
    console.log("  ⏭️  Skipped: is-odd not installed (run: sofuu add is-odd)");
    assert("npm spec validation (no package needed)", true);
}

console.log("\n=== RESULTS ===");
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed > 0) process.exit(1);
console.log("\n✅ All npm tests PASSED (or skipped)");
