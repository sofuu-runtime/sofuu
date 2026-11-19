// t36_best_effort_opt_in.js — R3.3: a best-effort path (mmq) is not the
// default; the default engine path is exact (cpu-tiles); and the opt-in knob
// reaches a child runtime process only when it is passed explicitly.
//
// PURE CHECK: it reads the manifest and the child's environment; it runs
// no model, tool or network exchange. The child environment is filtered to
// the runtime's allowlist, so the default child carries no opt-in knob.
//
// RESOURCE: pure: the capability manifest and the child's environment
//
// Run: ./sofuu run scripts/contract/harness/t36_best_effort_opt_in.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

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

async function main() {
    const raw = sofuu.capabilities();
    const caps = typeof raw === "string" ? P(raw) : raw;
    assert("capabilities returned valid engine paths", caps && caps.engine && caps.engine.paths);

    const mmq = caps.engine.paths["mmq"];
    assert("mmq path declared", Boolean(mmq));
    assert("mmq tier is best-effort", mmq && mmq.tier === "best-effort");
    assert("mmq is NOT default (default === false, opt-in only)", mmq && mmq.default === false);

    const cpuTiles = caps.engine.paths["cpu-tiles"];
    assert("cpu-tiles is default (default === true, exact path)", cpuTiles && cpuTiles.default === true && cpuTiles.tier === "exact");

    // The suite must exercise the default path: an opt-in knob exported in the
    // caller's environment would silently turn this case into a best-effort run.
    assert("the harness environment does not pre-set SOFUU_DECODE_MMQ",
        !process.env.SOFUU_DECODE_MMQ);

    const binary = process.env.SOFUU_BIN || "./sofuu";
    assert(`runtime binary '${binary}' exists`, sofuu.fs.existsSync(binary));
    const bin = `"${binary}"`;

    // Test environment flag gating:
    // With no env flag, SOFUU_DECODE_MMQ is unset
    const rDefault = await sh(`${bin} eval 'console.log(process.env.SOFUU_DECODE_MMQ || "unset")'`);
    assert("SOFUU_DECODE_MMQ is unset in the default child runtime process", rDefault.out.trim().indexOf("unset") >= 0);

    // With explicit flag, the opt-in is engaged
    const rOptIn = await sh(`SOFUU_DECODE_MMQ=1 ${bin} eval 'console.log(process.env.SOFUU_DECODE_MMQ)'`);
    assert("SOFUU_DECODE_MMQ=1 reaches the child runtime process when explicitly passed", rOptIn.out.trim() === "1");

    console.log(`\n=== t36 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t36:", (e && e.message) || e);
    process.exit(1);
});
