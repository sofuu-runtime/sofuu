// tests/ai_stream_test.js — ai.stream test (assert-based, A0.2)
// Requires a live provider. Tests streaming via mock.

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== ai.stream() Test ===\n");

assert("sofuu.ai.stream is function", typeof sofuu.ai.stream === "function");

async function main() {
    // Mock ai.stream to test the async iterator protocol
    const origStream = sofuu.ai.stream;
    sofuu.ai.stream = function(opts) {
        const chunks = ["Hello", " ", "world", "!"];
        const stream = {
            [Symbol.asyncIterator]() {
                let i = 0;
                return {
                    next() {
                        if (i < chunks.length) {
                            return Promise.resolve({ value: { text: chunks[i++] }, done: false });
                        }
                        return Promise.resolve({ value: undefined, done: true });
                    }
                };
            },
            usage: { promptTokens: 2, completionTokens: 4 }
        };
        return stream;
    };

    const stream = sofuu.ai.stream({
        messages: [{ role: "user", content: "Say hello" }],
        provider: "mock",
        model: "mock-model"
    });

    let full = "";
    let chunkCount = 0;
    for await (const chunk of stream) {
        assert("chunk has text", typeof chunk.text === "string");
        full += chunk.text;
        chunkCount++;
    }

    assert("received 4 chunks", chunkCount === 4);
    assert("full text === 'Hello world!'", full === "Hello world!");
    assert("stream.usage exists", stream.usage !== undefined);

    // Restore
    sofuu.ai.stream = origStream;

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All ai.stream tests PASSED (mock provider)");
}

main().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
