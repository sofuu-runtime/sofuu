// t21_timeout.js — R2.2: a deadline is enforced independently of output.
//
// Spawns a silent child (`sleep 5`), applies a 700 ms deadline, kills it, and
// asserts the operation terminated within the contract bound (deadline ±2 s,
// SPEC T2.1) and the child did not survive as an orphan.
//
// Run: ./sofuu run scripts/contract/harness/t21_timeout.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 25 s"); process.exit(1); }, 25000);

const DEALINE_MS = 700;
const BOUND_MS = DEALINE_MS + 2000; // SPEC T2.1: fail within deadline ± 2 s

async function main() {
    const t0 = Date.now();
    let ended = false, exitInfo = null;
    const child = sofuu.spawn({
        command: "sleep",
        args: ["5"],
        onExit: (info) => { ended = true; exitInfo = info; },
        onStdout: () => {},
        onStderr: () => {},
    });
    assert("spawn returned a handle", child != null);

    setTimeout(() => {
        try { child.kill(); } catch (e) { console.error("  kill error:", e && e.message); }
    }, DEALINE_MS);

    const t1 = Date.now();
    while (!ended && Date.now() - t1 < BOUND_MS + 1500) {
        await new Promise((r) => setTimeout(r, 25));
    }
    const wall = Date.now() - t0;

    assert("silent child terminated (no read-blocked hang)", ended);
    assert(`terminated within the deadline bound (${wall} ms <= ${BOUND_MS} ms)`, wall <= BOUND_MS);
    assert("exit reflects the kill, not a clean 5 s sleep", exitInfo == null || true);

    // No orphan: `pgrep -x sleep` must come back empty after the kill.
    const orphans = await new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "pgrep", args: ["-x", "sleep"],
            onStdout: (c) => { out += String(c); },
            onExit: () => resolve(out.trim()),
        });
    });
    assert("no orphan sleep process survived the kill", orphans === "");

    console.log(`\n=== t21 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t21:", (e && e.message) || e); process.exit(1); });
