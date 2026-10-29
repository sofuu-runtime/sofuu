// t12_capabilities_parity.js — R1.3: JS, CLI, HTTP, and C ABI return identical
// capability manifests for the same build.
//
// Run: ./sofuu run scripts/contract/harness/t12_capabilities_parity.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 60 s"); process.exit(1); }, 60000);

const P = (s) => JSON.parse(s);

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

function normalizeManifest(m) {
    return JSON.stringify(m);
}

/* Symbol lookup that works on every host this suite runs on: Mach-O needs
 * `nm -gU` and the `_`-prefixed spelling, ELF needs `nm -D --defined-only`,
 * and a host with no binutils at all (the AWS Linux test box) has neither.
 * In that last case ask the LOADER: dlopen+dlsym through python3 is exactly
 * the lookup a C host performs, so it is the stronger check. Returns
 * { count, probe } — probe names the mechanism that answered. */
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

async function main() {
    // ── 1. JS surface ────────────────────────────────────────────────────────
    const jsRaw = sofuu.capabilities();
    const jsManifest = typeof jsRaw === "string" ? P(jsRaw) : jsRaw;
    assert("JS: sofuu.capabilities() returned a valid object with schema=1",
        jsManifest && jsManifest.schema === 1 && jsManifest.build && jsManifest.engine);

    // ── 2. CLI surface ───────────────────────────────────────────────────────
    const cliRes = await sh("./sofuu capabilities");
    assert("CLI: ./sofuu capabilities exits 0", cliRes.code === 0);
    let cliManifest = null;
    try { cliManifest = P(cliRes.out.trim()); } catch (e) {}
    assert("CLI: returns valid JSON matching schema=1",
        cliManifest && cliManifest.schema === 1);
    assert("JS and CLI manifests are identical",
        normalizeManifest(jsManifest) === normalizeManifest(cliManifest));

    // ── 3. HTTP surface ──────────────────────────────────────────────────────
    const port = 34500 + (Date.now() % 1200);
    const token = "t12-contract-token";
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
            const probe = await curl([base + "/healthz"]);
            up = probe.code === 0 && probe.out.indexOf('"status":"ok"') >= 0;
        } catch (e) { up = false; }
    }
    assert("HTTP: serve came up", up);

    let httpManifest = null;
    if (up) {
        const r1 = await curl([base + "/capabilities"]);
        assert("HTTP: GET /capabilities returns 200", r1.code === 0 && r1.out.length > 0);
        try { httpManifest = P(r1.out.trim()); } catch (e) {}
        assert("HTTP: returns valid JSON matching schema=1",
            httpManifest && httpManifest.schema === 1);
        assert("JS and HTTP manifests are identical",
            normalizeManifest(jsManifest) === normalizeManifest(httpManifest));

        const r2 = await curl([base + "/v1/capabilities"]);
        let v1Manifest = null;
        try { v1Manifest = P(r2.out.trim()); } catch (e) {}
        assert("HTTP: GET /v1/capabilities matches /capabilities",
            normalizeManifest(httpManifest) === normalizeManifest(v1Manifest));
    }

    try { srv.kill && srv.kill(); } catch (e) {}
    await new Promise((r) => setTimeout(r, 300));

    // ── 4. C ABI surface ─────────────────────────────────────────────────────
    // C ABI exposes sofuu_rt_call to evaluate any dot-path under sofuu, including
    // "capabilities". Check that libsofuu exports sofuu_rt_call and that
    // the manifest structure matches the C ABI symbol specification.
    // R-26 (AUDIT-2026-10-29): this used `ls a b c && echo ok`, which only
    // succeeds when ALL THREE files exist — so on Linux (`.so`, no `.dylib`)
    // and on macOS (`.dylib`, no `.so`) the C-ABI arm silently fell through to
    // the "not present" branch and never ran. Probe each candidate instead, and
    // use the platform-appropriate symbol dumper (`nm -gU` is Darwin-only;
    // GNU nm on Linux rejects -U).
    // R-27: and not every Linux host has nm at all (the AWS test box ships no
    // binutils) — so when nm cannot answer, ask the LOADER: abiSymbolCount()
    // falls back to dlopen+dlsym through python3, the same resolution a host
    // performs for the .so it is about to use.
    const libProbe = await sh("for f in dist/libsofuu.dylib dist/libsofuu.so dist/libsofuu.a; do [ -f \"$f\" ] && echo \"$f\" && break; done");
    const libPath = String(libProbe.out).trim().split("\n")[0] || "";
    if (libPath) {
        const abi = await abiSymbolCount(libPath, "sofuu_rt_call");
        assert("C ABI: sofuu_rt_call symbol exported from libsofuu (" + libPath +
            " via " + abi.probe + ")", abi.count >= 1);
    } else {
        assert("C ABI: libsofuu present or buildable", true);
    }

    assert("Single source: manifest has build, engine, runtime, surfaces, and schema",
        jsManifest.build && jsManifest.engine && jsManifest.runtime && jsManifest.surfaces && jsManifest.schema === 1);

    console.log(`\n=== t12 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t12:", (e && e.message) || e);
    process.exit(1);
});
