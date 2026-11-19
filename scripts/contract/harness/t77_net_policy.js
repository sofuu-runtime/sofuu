// t77_net_policy.js — R7.1/R7.2 (net dimension): wildcard host patterns match
// only the intended suffix, and `:port`, `user@`, and `[v6]` normalize before
// the match — the same decision on the JS and CLI surfaces, and a denied host
// never reaches the network.
//
// The evaluator checks are pure policy decisions (no backend involved). The
// fetch check is a REAL request attempt: a denied host must be refused by the
// policy itself, with the rule id, before any name is resolved — a
// transport failure would mean the policy did not gate the request.
//
// RESOURCE: pure: net policy evaluated in process; the denied fetch is refused before any name is resolved
//
// Run: ./sofuu run scripts/contract/harness/t77_net_policy.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
/* The CLI surface is a separate process per decision, and the shipped binary
 * spends seconds in start-up on some hosts — the budget covers four of them. */
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 120 s"); process.exit(1); }, 120000);

const P = (s) => JSON.parse(s);

/* The binary under test — the one running this harness. The CLI surface must
 * be the same build as the JS one. */
const BIN = String((typeof process !== "undefined" && process.argv && process.argv[0]) || "./sofuu");

function ev(req) {
    return P(sofuu.policy.evaluate(JSON.stringify(req)));
}

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({ command: "/bin/sh", args: ["-c", cmd],
            onStdout: (c) => { out += String(c); }, onExit: (c) => { code = c; resolve({ code, out }); } });
    });
}

/* The CLI evaluator for one host — the same policy object, the same
 * evaluator, the same normalization. */
async function cliWhy(host) {
    const c = await sh(BIN + " policy why net.fetch host='" + host + "' --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ net: { deny: ["*.evil.example"] } }) + "\nEOP");
    return P(String(c.out).trim().split("\n").pop());
}

async function main() {
    sofuu.policy.set(JSON.stringify({ net: { deny: ["*.evil.example"] } }));

    // ── suffix anchoring ─────────────────────────────────────────────────
    assert("deny: the pattern itself",
        ev({ op: "net.fetch", host: "sub.evil.example" }).decision === "deny");
    assert("deny: nested under the suffix",
        ev({ op: "net.fetch", host: "a.b.evil.example" }).decision === "deny");
    assert("allow: an unrelated host",
        ev({ op: "net.fetch", host: "good.example.org" }).decision === "allow");
    assert("allow: a SUSPICIOUS lookalike that the pattern does not cover",
        ev({ op: "net.fetch", host: "evil.example.fake" }).decision === "allow");
    assert("allow: bare TLD string not matched inside a longer name",
        ev({ op: "net.fetch", host: "notevil.example.org" }).decision === "allow");

    // ── normalization before the match ───────────────────────────────────
    assert("normalize: :port stripped", 
        ev({ op: "net.fetch", host: "sub.evil.example:8443" }).decision === "deny");
    assert("normalize: user@ stripped",
        ev({ op: "net.fetch", host: "agent@sub.evil.example" }).decision === "deny");
    assert("normalize: [v6] literal still judged by its host",
        ev({ op: "net.fetch", host: "[2001:db8::1]" }).decision === "allow");
    assert("allow: URL-ish form denied when the host inside matches",
        ev({ op: "net.fetch", host: "https://user@sub.evil.example:443/path?q=1" }).decision === "deny");

    // ── the same decision through the CLI (one evaluator, no drift) ──────
    const c1 = await cliWhy("sub.evil.example:8443");
    assert("CLI: the identical deny + rule",
        c1.decision === "deny" && String(c1.rule).indexOf("net") >= 0);
    const c2 = await cliWhy("good.example.org");
    assert("CLI: the identical allow",
        c2.decision === "allow");
    const c3 = await cliWhy("https://user@sub.evil.example:443/path?q=1");
    assert("CLI: the URL-ish form normalizes to the same deny",
        c3.decision === "deny" && String(c3.rule).indexOf("net") >= 0);
    const c4 = await cliWhy("[2001:db8::1]");
    assert("CLI: the [v6] literal reaches the same allow",
        c4.decision === "allow");

    // ── enforcement: a denied host never reaches the network ────────────
    // A real request attempt. The refusal must come from the policy (typed,
    // with the rule id); a transport failure would mean the gate did not
    // run and the name was resolved anyway.
    let netErr = null;
    try { await fetch("http://sub.evil.example/t77-probe"); }
    catch (e) { netErr = e; }
    assert("a real request to the denied host is refused by the policy, with its rule",
        !!netErr && String(netErr.code) === "E_PERMISSION_DENIED" &&
        String(netErr.rule) === "R7.1/net.deny");
    assert("the refusal is the policy's, not a transport failure (no name was resolved)",
        !!netErr && String(netErr.message || "").indexOf("denied by policy") >= 0);

    sofuu.policy.set("{}");
    console.log(`\n=== t77 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t77:", (e && e.message) || e); process.exit(1); });
