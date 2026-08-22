// tests/sse_test.js — SSE parser test (assert-based, A0.2)

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== SSE Parser Test ===\n");

assert("sofuu.SSEParser is function", typeof sofuu.SSEParser === "function");

const sse = new sofuu.SSEParser();

let messages = [];
sse.onMessage = (evt, data) => {
    messages.push({ event: evt, data: data });
};

sse.feed("data: Hello World");
sse.feed("\n\n");
sse.feed("data: ChatGPT\n\n");
sse.feed("event: custom\ndata: line1\ndata: line2\n\n");

assert("first message data === 'Hello World'", messages.length >= 1 && messages[0].data === "Hello World");
assert("second message data === 'ChatGPT'",   messages.length >= 2 && messages[1].data === "ChatGPT");
assert("third message event === 'custom'",     messages.length >= 3 && messages[2].event === "custom");
assert("third message multi-line data",       messages.length >= 3 && messages[2].data === "line1\nline2");

console.log("\n=== RESULTS ===");
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed > 0) process.exit(1);
console.log("\n✅ All SSE tests PASSED");
