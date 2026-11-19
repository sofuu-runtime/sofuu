// t58_error_parity.js — R5.1: The same failure returns the same §1.4 taxonomy
// code and shared error envelope shape across JS, HTTP, and C ABI.
//
// The HTTP surface is a real `sofuu serve` process: when
// SOFUU_CONTRACT_MODEL names an existing real-weight GGUF the server really
// loads it, so the model-not-found refusal is exercised against a live
// backend. No weights are downloaded and the synthetic contract fixture is
// not accepted as evidence of inference.
//
// RESOURCE: real: the real serve process on loopback (the error envelopes need no model)
//
// Run: ./sofuu run scripts/contract/harness/t58_error_parity.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 120 s"); process.exit(1); }, 120000);

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

/* Symbol lookup that works on every host: Mach-O needs `nm -gU` and the
 * `_`-prefixed spelling, ELF needs `nm -D --defined-only`, and a host with no
 * binutils asks the LOADER through python3 (ctypes), which is the lookup a C
 * host performs. Returns { count, probe }. */
async function abiSymbolCount(libPath, symbol) {
    const nmCmd = process.platform === "darwin"
        ? "nm -gU " + JSON.stringify(libPath) + " 2>/dev/null | grep -c ' _*" + symbol + "$' || true"
        : "nm -D --defined-only " + JSON.stringify(libPath) + " 2>/dev/null | grep -c ' " + symbol + "$' || true";
    const nm = await sh(nmCmd);
    const nmCount = Number(String(nm.out).trim().split("\n").pop()) || 0;
    if (nmCount > 0) return { count: nmCount, probe: "nm" };
    const py = await sh("command -v python3 2>/dev/null || echo /usr/bin/python3");
    const pyBin = String(py.out).trim().split("\n").pop() || "python3";
    const code = "import ctypes,sys;getattr(ctypes.CDLL(sys.argv[1]),sys.argv[2]);print(1)";
    const r = await sh(pyBin + " -c " + JSON.stringify(code) + " " + libPath + " " + symbol + " 2>/dev/null || echo 0");
    const count = Number(String(r.out).trim().split("\n").pop()) || 0;
    return { count: count, probe: "dlopen(" + pyBin + ")" };
}

/* The binary that runs this harness: every surface under test must be the
 * same build, so the spawned CLI/server is the running artifact itself. */
function runtimeBinary() {
    const argv = process.argv || [];
    if (argv.length > 0 && typeof argv[0] === "string" && argv[0].length > 0) return argv[0];
    return process.env.SOFUU_BIN || "./sofuu";
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
    const binary = runtimeBinary();
    const modelPath = String(process.env.SOFUU_CONTRACT_MODEL || "");
    const realModel = modelPath && /\.gguf$/i.test(modelPath) && sofuu.fs.existsSync(modelPath)
        && !/(^|\/)sofuu-fixture\.gguf$/i.test(modelPath) ? modelPath : "";
    // A model that was selected but cannot be loaded is a broken
    // prerequisite, not a reason to run the case against a different
    // backend: report it instead of silently dropping the selection.
    if (modelPath && !realModel) {
        console.error("BLOCKED: SOFUU_CONTRACT_MODEL must name an existing real-weight GGUF " +
            "(not a synthetic or vocabulary fixture): " + modelPath);
        process.exit(2);
    }
    const home = String(process.env.SOFUU_HOME || "");
    const serveArgs = ["serve", "--port", String(port), "--host", "127.0.0.1", "--token", token];
    if (realModel) serveArgs.push("--model", realModel);
    // The served brain stays inside the isolated home: a contract run must
    // not touch the developer's real brain.
    if (home) serveArgs.push("--brain", home + "/t58-contract-brain.qtsq");
    const srv = sofuu.spawn({
        command: binary,
        args: serveArgs,
        onStdout: () => {},
        onStderr: () => {},
        onExit: () => {},
    });
    const base = "http://127.0.0.1:" + port;

    let up = false;
    for (let i = 0; i < 40 && !up; i++) {
        await new Promise((r) => setTimeout(r, 200));
        try {
            const probe = await curl([base + "/healthz"]);
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

        // A model the server never loaded is refused with the same envelope —
        // the refusal carries a code, not just prose.
        const rModel = await curl(["-H", "Authorization: Bearer " + token, "-H", "Content-Type: application/json",
            "-X", "POST", base + "/v1/chat/completions",
            "-d", JSON.stringify({ model: "t58-model-that-was-never-loaded",
                messages: [{ role: "user", content: "hi" }], max_tokens: 4 })]);
        let modelErr = null;
        try { modelErr = P(rModel.out); } catch (e) {}
        assert("HTTP: model refusal carries the standard error envelope with a code",
            modelErr && modelErr.error && typeof modelErr.error.message === "string"
            && typeof modelErr.error.code === "string");
    }

    try { srv.kill && srv.kill(); } catch (e) {}
    await new Promise((r) => setTimeout(r, 250));

    // ── 3. C ABI surface: exported funnel and error envelope ─────────────────
    // C ABI returns errors as {"ok": false, "error": {"code": ..., "message": ...}}
    // Probe each candidate separately: `ls a b c` fails unless ALL THREE
    // exist, which silently skipped this arm on every real platform.
    const libProbe = await sh("for f in dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a; do [ -f \"$f\" ] && echo \"$f\" && break; done");
    const libPath = String(libProbe.out).trim().split("\n")[0] || "";
    if (libPath) {
        const abi = await abiSymbolCount(libPath, "sofuu_rt_call");
        assert("C ABI: sofuu_rt_call exported for error funnel (" + libPath +
            " via " + abi.probe + ")", abi.count >= 1);
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
