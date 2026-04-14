// examples/npm_test.js — Testing npm package resolution & CJS shim
// Run:
//   sofuu add is-odd is-number
//   sofuu run examples/npm_test.js

import isOdd from 'is-odd';

console.log("\n=== npm module & CJS shim test ===\n");

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else       { console.error("  ❌ FAIL: " + label); results.failed++; }
}

assert("isOdd(3) === true", isOdd(3) === true);
assert("isOdd(4) === false", isOdd(4) === false);
assert("isOdd(11) === true", isOdd(11) === true);

console.log(`\n=== RESULTS ===`);
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed === 0) {
    console.log("\n✅ npm CJS modules — all tests PASSED\n");
} else {
    console.error(`\n❌ ${results.failed} test(s) failed`);
    process.exit(1);
}
