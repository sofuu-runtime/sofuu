// tests/fetch_test.js — fetch() Response API test (assert-based, A0.2).
// P2-26 (AUDIT-2026-09-01): this used to hit httpbin.org and swallow every
// failure as "Skipped" (a fully broken fetch passed as "API surface
// exists"). It now runs a LOCAL in-process mock server (real curl path,
// no network, no keys) with hard assertions — the same pattern as
// ai_stream_error_frame_test.js. Covers: status/ok/headers/text/json/
// 404 handling, chunked bodies, and POST request-body delivery.

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== fetch() Test ===\n");

assert("sofuu.fetch is function", typeof sofuu.fetch === "function");

const JSON_1 = JSON.stringify({ url: "/get", hello: "world" });
const JSON_404 = JSON.stringify({ error: "not found" });
const JSON_OK = JSON.stringify({ status: "ok" });

/* POST-body evidence (mock_llm_server pattern: request data drives
 * decisions, responses are constants — nothing from the request is ever
 * reflected back into a response). */
let lastEchoLen = -1;

const handler = (req, res) => {
    if (req.url === "/get") {
        res.writeHead(200, { "Content-Type": "application/json", "X-Mock": "yes" });
        res.end(JSON_1);
        return;
    }
    if (req.url === "/chunked") {
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.write("chunk-one ");
        res.write("chunk-two");
        res.end();
        return;
    }
    if (req.url === "/echo") {
        try { lastEchoLen = String(req.body || "").length; } catch (e) { lastEchoLen = -1; }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON_OK);
        return;
    }
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON_404);
};

async function main() {
    let server = null, port = 0;
    const base = 24400 + (Date.now() % 4000);
    for (let i = 0; i < 6; i++) {
        const p = base + i * 131;
        try {
            server = sofuu.createServer(handler);
            server.listen(p, "127.0.0.1");
            port = p;
            break;
        } catch (e) { server = null; }
    }
    if (!server) { console.log("FAIL could not bind mock port"); process.exit(1); }
    /* Let the listener actually start accepting before the first fetch.
     * listen() returns before the socket is servicing connections, and this
     * server runs in-process on the same event loop as the client. Firing
     * fetch() in the same tick was a race: macOS lost it and passed, the
     * Linux runners won it and every later fetch in the file failed with an
     * opaque "Test failed: {}". One loop turn is enough to settle it. */
    await new Promise((r) => setTimeout(r, 50));
    const URL = "http://127.0.0.1:" + port;

    // ── GET + Response surface ─────────────────────────────────────
    const res = await sofuu.fetch(URL + "/get");
    assert("res.status is number", typeof res.status === "number");
    assert("res.status === 200", res.status === 200);
    assert("res.ok === true", res.ok === true);
    assert("res.headers exists", typeof res.headers === "object");

    const hdr = res.headers && (res.headers.get ? res.headers.get("content-type") : null);
    assert("res.headers.get('content-type') === 'application/json'", String(hdr) === "application/json");

    const text = await res.text();
    assert("res.text() returns string", typeof text === "string");
    assert("res.text() has content", text.length > 0);

    // ── json() parses the body ────────────────────────────────────
    const resJ = await sofuu.fetch(URL + "/get");
    const data = await resJ.json();
    assert("res.json() parses object", data && typeof data === "object");
    assert("res.json() carries payload", data.hello === "world");

    // ── 404 path ──────────────────────────────────────────────────
    const res2 = await sofuu.fetch(URL + "/nope");
    assert("404 status correct", res2.status === 404);
    assert("404 ok === false", res2.ok === false);

    // ── chunked body arrives whole ────────────────────────────────
    const resC = await sofuu.fetch(URL + "/chunked");
    const ctext = await resC.text();
    assert("chunked body concatenated", ctext === "chunk-one chunk-two");

    // ── POST body reaches the server ───────────────────────────────
    lastEchoLen = -1;
    const resP = await sofuu.fetch(URL + "/echo", { method: "POST", body: "PING-PAYLOAD" });
    await resP.text();
    assert("POST body arrived intact (server saw the full length)",
        lastEchoLen === "PING-PAYLOAD".length);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All fetch tests PASSED");
}

main().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
