// tests/abi_script_test.js — regression net for scripts/check_abi.sh.
//
// The ABI gate was red for three separate reasons, each of which made it
// useless rather than strict:
//
//   1. `sofuu_voice_*` was exported and BASELINED but missing from the
//      PUBLIC_RE allowlist. A symbol the filter hides can never be
//      detected as added OR removed, so the gate could not see it at all
//      and then reported it as REMOVED from a library that exports it.
//   2. `comm` is byte-order and needs both inputs in the same collation.
//      The extracted list used locale `sort` and the baseline was ASCII,
//      so under the C locale the orders diverged and comm invented a
//      phantom removal of `sofuu_run_jobs`.
//   3. `$SOFUU_ABI_ALLOW_NEW` was read unguarded under `set -u`, so the
//      "acknowledged addition" path CRASHED (unbound variable) instead
//      of reporting the failure it was written to report.
//
// A gate that cries wolf, or crashes, or is blind to a whole prefix, gets
// ignored — and an ignored ABI gate protects nothing. These assertions
// run the real script against the real library and check the real files.
//
// Run:  ./sofuu run tests/abi_script_test.js   (needs `make libsofuu` first)

const results = { passed: 0, failed: 0 };
function assert(label, cond, detail) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label + (detail ? "\n     " + detail : "")); results.failed++; }
}

console.log("=== ABI check script audit ===\n");

const ROOT = process.env.SOFUU_REPO || ".";
const script = await sofuu.fs.readFile(ROOT + "/scripts/check_abi.sh");
const baseline = await sofuu.fs.readFile(ROOT + "/scripts/abi_symbols.txt");

// ── 1. allowlist vs baseline coverage ─────────────────────────────
// Every public prefix in the baseline must be matched by PUBLIC_RE, or
// the symbol is invisible to the gate.
const pubRe = script.match(/PUBLIC_RE='([^']+)'/);
assert("PUBLIC_RE is defined", !!pubRe);
const re = new RegExp(pubRe ? pubRe[1] : "$^");

const symbols = baseline.split("\n").map(s => s.trim()).filter(Boolean);
/* The allowlist mixes prefixes (sofuu_rt_) with whole names
 * (sofuu_run_jobs). What matters is that every BASELINED symbol matches
 * it — a symbol the filter hides can be neither detected as added nor as
 * removed, which is bug #1. */
const invisible = symbols.filter(s => !re.test(s));
assert("every baselined symbol is visible to PUBLIC_RE", invisible.length === 0,
    invisible.length ? "invisible: " + invisible.join(", ") : "");
assert("sofuu_voice_ is in the allowlist", re.test("sofuu_voice_speak"));
assert("the baselined voice symbols exist", symbols.indexOf("sofuu_voice_speak") >= 0 &&
    symbols.indexOf("sofuu_voice_transcribe") >= 0);

// ── 2. comm collation ─────────────────────────────────────────────
assert("both comm calls are byte-ordered (LC_ALL=C)",
    /LC_ALL=C comm -13/.test(script) && /LC_ALL=C comm -23/.test(script),
    "comm compares byte order; mismatched collation invents removals");
assert("the baseline is sorted with the same collation",
    /LC_ALL=C sort -u "\$BASELINE"/.test(script));
/* The actual failure mode, reproduced directly: sorting the baseline under
 * the C locale and diffing must list nothing as missing when the library
 * really does export it. */
const sorted = symbols.slice().sort();
const joined = sorted.join("\n");
const unsortedOrder = symbols.join("\n");
assert("baseline is already in byte order", joined === unsortedOrder,
    "baseline not byte-sorted; comm would mis-order the two lists");

// ── 3. set -u safety ──────────────────────────────────────────────
const unguarded = script.match(/if \[ "\$SOFUU_ABI_ALLOW_NEW" = /);
assert("SOFUU_ABI_ALLOW_NEW is read with a default", !unguarded,
    'under `set -u` an unset $SOFUU_ABI_ALLOW_NEW aborts the script');
assert("the default is used", /\$\{SOFUU_ABI_ALLOW_NEW:-0\}/.test(script));

// ── 4. the symbol that caused the phantom break ───────────────────
assert("sofuu_run_jobs is baselined", symbols.indexOf("sofuu_run_jobs") >= 0);
assert("sofuu_run_jobs is in the allowlist", re.test("sofuu_run_jobs"));
assert("sofuu_loop_shutdown_engine is baselined (it is a real export)",
    symbols.indexOf("sofuu_loop_shutdown_engine") >= 0,
    "rt/loop.rs:130 exports it; an unbaselined export fails the gate every run");

// ── 5. the library, if built, agrees with the baseline ────────────
/* Run the REAL gate rather than re-implementing nm parsing in JS: the
 * point is that `make abi-check` exits 0, which is the property CI
 * depends on. */
for (const p of ["/dist/libsofuu.dylib", "/dist/libsofuu.so"]) {
    if (!(await sofuu.fs.exists(ROOT + p))) continue;
    let code = -1, out = "";
    sofuu.spawn({
        command: "bash",
        args: [ROOT + "/scripts/check_abi.sh", ROOT + p],
        cwd: ROOT,
        onStdout: (d) => { out += String(d); },
        onStderr: (d) => { out += String(d); },
        onExit: (c) => { code = c; },
    });
    const t0 = Date.now();
    while (code === -1 && Date.now() - t0 < 60000) {
        await new Promise(r => setTimeout(r, 100));
    }
    assert("the real abi gate exits 0 on " + p, code === 0,
        "exit " + code + "\n" + out.split("\n").slice(0, 8).join("\n     "));
    assert("the gate confirms the baseline matches", /match baseline exactly/.test(out),
        out.split("\n").slice(0, 6).join("\n     "));
    break;
}

console.log("\n=== RESULTS ===");
console.log("Passed: " + results.passed + " | Failed: " + results.failed);
if (results.failed > 0) process.exit(1);
console.log("\n✅ ABI gate is sound (allowlist, collation, set -u, baseline)");
