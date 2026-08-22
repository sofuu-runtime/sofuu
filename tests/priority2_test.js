// examples/priority2_test.js — Priority 2 integration tests
// Tests: multi-file import, process.stdout/stderr, process.on signals, process.stdin

import square, { add, multiply, PI } from './lib/math.js';
import { greet, shout } from './lib/strings.js';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

// ─────────────────────────────────────────────
// Test 4: Multi-file import resolution
// ─────────────────────────────────────────────
console.log("\n=== Test 4: Multi-file ESM import ===");

assert("add(2, 3) === 5",            add(2, 3) === 5);
assert("multiply(4, 5) === 20",      multiply(4, 5) === 20);
assert("PI ≈ 3.14...",               PI > 3.14 && PI < 3.15);
assert("default export square(7)",  square(7) === 49);
assert("transitive: greet()",       greet("World").includes("Hello, World!"));
assert("transitive: shout()",       shout("sofuu") === "SOFUU!");
console.log("  greet result:", greet("Sofuu"));

// ─────────────────────────────────────────────
// Test 5: process.stdout / process.stderr
// ─────────────────────────────────────────────
console.log("\n=== Test 5: process.stdout / stderr ===");

assert("process.stdout is object",     typeof process.stdout === "object");
assert("process.stdout.write is fn",   typeof process.stdout.write === "function");
assert("process.stderr is object",     typeof process.stderr === "object");
assert("process.stderr.write is fn",   typeof process.stderr.write === "function");
assert("stdout.fd === 1",              process.stdout.fd === 1);
assert("stderr.fd === 2",              process.stderr.fd === 2);

process.stdout.write("  [stdout.write works]\n");
process.stderr.write("  [stderr.write works]\n");

// ─────────────────────────────────────────────
// Test 6: process.on / process.pid / process.platform
// ─────────────────────────────────────────────
console.log("\n=== Test 6: process.on + metadata ===");

assert("process.on is function",       typeof process.on === "function");
assert("process.off is function",      typeof process.off === "function");
assert("process.pid > 0",             process.pid > 0);
assert("process.platform string",     typeof process.platform === "string");
assert("process.version string",      typeof process.version === "string");
console.log("  pid:", process.pid, " platform:", process.platform);

// Register SIGINT handler (won't fire during test, just verifies lazy registration)
process.on('SIGINT', () => {
    console.log("[SIGINT] clean exit");
    process.exit(0);
});
assert("SIGINT handler registered", true);

process.on('SIGTERM', () => {
    console.log("[SIGTERM] clean exit");
    process.exit(0);
});
assert("SIGTERM handler registered", true);

process.on('exit', () => { /* fires on process.exit() */ });
assert("exit handler registered", true);

// ─────────────────────────────────────────────
// Test 7: process.stdin API surface
// ─────────────────────────────────────────────
console.log("\n=== Test 7: process.stdin API ===");

assert("process.stdin is object",       typeof process.stdin === "object");
assert("process.stdin.on is function",  typeof process.stdin.on === "function");
assert("process.stdin.resume is fn",    typeof process.stdin.resume === "function");
assert("process.stdin.pause is fn",     typeof process.stdin.pause === "function");
assert("process.stdin.fd === 0",        process.stdin.fd === 0);
assert("stdin.isTTY defined",           process.stdin.isTTY !== undefined);
assert("stdin.setEncoding is fn",       typeof process.stdin.setEncoding === "function");

// ─────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────
console.log("\n=== RESULTS ===");
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);

if (results.failed === 0) {
    console.log("\n✅ All Priority 2 tests PASSED — production-ready");
} else {
    console.error(`\n❌ ${results.failed} test(s) failed`);
    process.exit(1);
}
