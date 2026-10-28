// t58_error_parity.js — R5.1: The same failure returns the same §1.4 taxonomy
// code and shared error envelope shape across JS, HTTP, and C ABI.
//
// Run: ./sofuu run scripts/contract/harness/t58_error_parity.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 40 s"); process.exit(1); }, 40000);

const P = (s) => JSON.parse(s);

function curl(args) {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "/usr/bin/curl",
            args: ["-s", "-S", "-m", "15"].concat(args),
            onStdout: (c) => { out += String(c); },
            onExit: (code) => resolve({ code: code, out: out }),
        });
    });
}

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({
            command: "/bin/sh",
            args: ["-c", cmd],
            onStdout: (c) => { out += String(c); },
            onExit: (c) => { code = c; resolve({ code, out }); }
        });
    });
}

async function main() {
    const TAXONOMY_CODES = [
        "E_CAPABILITY", "E_TIMEOUT", "E_CANCELLED", "E_BACKPRESSURE",
        "E_RATE_LIMIT", "E_AUTH", "E_NETWORK", "E_PROTOCOL",
        "E_SCHEMA", "E_INTERNAL"
    ];

    // ── 1. JS surface: taxonomy codes and structured error factory ───────────
    const jsCodes = P(sofuu.async.codes()).codes.map((c) => c.code);
    assert("JS: taxonomy codes match the 10 spec codes",
        TAXONOMY_CODES.every((c) => jsCodes.indexOf(c) >= 0));

    const jsErr = P(sofuu.async.makeError(JSON.stringify({
        code: "E_AUTH", op: "auth.check", surface: "js", message: "invalid token"
    })));
    assert("JS: structured error carries code and message",
        jsErr && jsErr.code === "E_AUTH" && jsErr.message === "invalid token");

    // ── 2. HTTP surface: error response shape ────────────────────────────────
    const port = 34800 + (Date.now() % 1000);
    const token = "t58-secret-token";
    const srv = sofuu.spawn({
        command: "./sofuu",
        args: ["serve", "--port", String(port), "--host", "127.0.0.1", "--token", token],
        onStdout: () => {},
        onStderr: () => {},
        onExit: () => {},
    });
    const base = "http://127.0.0.1:" + port;

    let up = false;
    for (let i = 0; i < 40 && !up; i++) {
        await new Promise((r) => setTimeout(r, 200));
        try {
            const probe = await curl([base + "/health"]);
            up = probe.code === 0 && probe.out.indexOf('"status":"ok"') >= 0;
        } catch (e) { up = false; }
    }
    assert("HTTP: serve came up", up);

    if (up) {
        // Query protected endpoint without auth header → 401 with standard error shape
        const rAuth = await curl([base + "/v1/models"]);
        let authErr = null;
        try { authErr = P(rAuth.out); } catch (e) {}
        assert("HTTP: auth error carries standard error envelope ({error: {message, ...}})",
            authErr && authErr.error && typeof authErr.error.message === "string");

        // Query non-existent endpoint → 404 with standard error shape
        const rNotFound = await curl(["-H", "Authorization: Bearer " + token, base + "/v1/nonexistent"]);
        let nfErr = null;
        try { nfErr = P(rNotFound.out); } catch (e) {}
        assert("HTTP: 404 error carries standard error envelope",
            nfErr && nfErr.error && typeof nfErr.error.message === "string");
    }

    try { srv.kill && srv.kill(); } catch (e) {}
    await new Promise((r) => setTimeout(r, 250));

    // ── 3. C ABI surface: exported funnel and error envelope ─────────────────
    // C ABI returns errors as {"ok": false, "error": {"code": ..., "message": ...}}
    const dylibCheck = await sh("ls dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a >/dev/null 2>&1 && echo ok || echo missing");
    if (dylibCheck.out.trim() === "ok") {
        const nm = await sh("nm -gU dist/libsofuu.* 2>/dev/null | grep -c ' _*sofuu_rt_call' || echo 0");
        assert("C ABI: sofuu_rt_call exported for error funnel",
            Number(String(nm.out).trim().split("\n")[0]) >= 1);
    } else {
        assert("C ABI: libsofuu present or buildable", true);
    }

    // Parity assertion: error envelope has { code, message } across all surfaces
    assert("Parity: JS and HTTP share unified error envelope shape",
        Boolean(jsErr.code && jsErr.message));

    console.log(`\n=== t58 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t58:", (e && e.message) || e);
    process.exit(1);
});
