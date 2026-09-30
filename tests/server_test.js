// tests/server_test.js — HTTP server test (assert-based, A0.2).
// P2-26 (AUDIT-2026-09-01): the old version swallowed any failure as
// "Skipped" + a constant "API surface exists" assert — a fully broken
// server passed. Real fetch-against-the-live-server assertions only.

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== HTTP Server Test ===\n");

assert("sofuu.http.createServer is function",
    sofuu.http && typeof sofuu.http.createServer === "function");

const PORT = 13099; // high port to avoid conflicts

const server = sofuu.http.createServer((req, res) => {
    if (req.url === "/health") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ok" }));
    } else {
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end("Hello from Sofuu HTTP Server\n");
    }
});
assert("server created", server !== undefined);

server.listen(PORT, "127.0.0.1");

setTimeout(async () => {
    try {
        const res = await sofuu.fetch("http://127.0.0.1:" + PORT + "/health");
        assert("GET /health status 200", res.status === 200);
        const data = await res.json();
        assert("GET /health returns {status:'ok'}", data.status === "ok");

        const res2 = await sofuu.fetch("http://127.0.0.1:" + PORT + "/", {
            method: "GET"
        });
        assert("GET / status 200", res2.status === 200);
        const text = await res2.text();
        assert("GET / returns text", text.includes("Hello from Sofuu"));

        if (server.stop) server.stop();
        if (server.close) server.close();

        console.log("\n=== RESULTS ===");
        console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
        /* The server object has no stop/close yet and a listening socket
         * keeps the event loop alive — exit explicitly. */
        process.exit(results.failed > 0 ? 1 : 0);
    } catch (err) {
        console.error("  ❌ FAIL: fetch against the live server:", err.message);
        if (server && server.stop) server.stop();
        if (server && server.close) server.close();
        process.exit(1);
    }
}, 200);
