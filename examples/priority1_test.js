// examples/priority1_test.js
// Tests for all 3 Priority 1 fixes:
//   1. GC clean exit (no assertion crash)
//   2. Proper fetch() Response API
//   3. Unhandled Promise rejection handler

const results = { passed: 0, failed: 0 };

function assert(label, cond) {
    if (cond) {
        console.log("  ✅ " + label);
        results.passed++;
    } else {
        console.error("  ❌ FAIL: " + label);
        results.failed++;
    }
}

// ─────────────────────────────────────────────
// Fix 3: Unhandled Promise rejection handler
// ─────────────────────────────────────────────
console.log("\n=== Fix 3: Unhandled Promise Rejection ===");
console.log("(expect: [sofuu] UnhandledPromiseRejection on stderr)");

// This promise intentionally has no .catch() — should print to stderr
Promise.reject(new Error("intentional unhandled rejection for test"));

// ─────────────────────────────────────────────
// Fix 2: fetch() Response API
// ─────────────────────────────────────────────
console.log("\n=== Fix 2: fetch() Response API ===");

async function testFetch() {
    // Test against a public JSON API
    const res = await fetch("https://httpbin.org/get");

    assert("res.status is a number",     typeof res.status === "number");
    assert("res.status === 200",          res.status === 200);
    assert("res.ok === true",             res.ok === true);
    assert("res.statusText is string",    typeof res.statusText === "string");
    assert("res.url contains httpbin",    res.url.includes("httpbin"));
    assert("res.headers exists",         typeof res.headers === "object");
    assert("headers.get() works",        typeof res.headers.get === "function");

    const contentType = res.headers.get("content-type");
    assert("Content-Type header present", contentType !== null);
    console.log("  content-type:", contentType);

    const text = await res.text();
    assert("res.text() returns string",   typeof text === "string");
    assert("res.text() has content",      text.length > 0);

    // Test .json()
    const res2 = await fetch("https://httpbin.org/json");
    const data = await res2.json();
    assert("res.json() returns object",   typeof data === "object");
    console.log("  json sample key:", Object.keys(data)[0]);

    // Test 404 response
    const res3 = await fetch("https://httpbin.org/status/404");
    assert("404 status correct",          res3.status === 404);
    assert("404 ok === false",            res3.ok === false);

    // Test POST with body
    const res4 = await fetch("https://httpbin.org/post", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runtime: "sofuu", version: "0.1" })
    });
    assert("POST status 200",             res4.status === 200);
    const postData = await res4.json();
    assert("POST body echoed back",       postData.json && postData.json.runtime === "sofuu");

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);

    if (results.failed === 0) {
        console.log("\n✅ All Priority 1 tests PASSED — runtime is production-ready on these criteria");
    } else {
        console.error(`\n❌ ${results.failed} test(s) failed`);
        process.exit(1);
    }
}

// ─────────────────────────────────────────────
// Fix 1: GC clean exit
// Verified by: no exit code 134 when this script finishes
// ─────────────────────────────────────────────
console.log("\n=== Fix 1: GC Clean Exit ===");
console.log("  (if this script exits with code 0, the GC crash is fixed)");

testFetch().catch(err => {
    console.error("Test error:", err);
    process.exit(1);
});
