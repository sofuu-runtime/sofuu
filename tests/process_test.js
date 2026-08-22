// tests/process_test.js — process bindings test (assert-based, A0.2)

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== Process Module Test ===\n");

assert("process.argv is array",     Array.isArray(process.argv));
assert("process.version is string",  typeof process.version === "string");
assert("process.platform is string", typeof process.platform === "string");
assert("process.env is object",      typeof process.env === "object");
assert("process.env.HOME exists",   typeof process.env.HOME === "string" || process.env.HOME === undefined);
assert("process.env.PATH is string", typeof process.env.PATH === "string");

console.log("  argv:", process.argv);
console.log("  version:", process.version);
console.log("  platform:", process.platform);

// Console methods
assert("console.log is function",   typeof console.log === "function");
assert("console.warn is function",  typeof console.warn === "function");
assert("console.error is function", typeof console.error === "function");
assert("console.info is function",  typeof console.info === "function");

console.log("\n=== RESULTS ===");
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed > 0) process.exit(1);
console.log("\n✅ All process tests PASSED");
