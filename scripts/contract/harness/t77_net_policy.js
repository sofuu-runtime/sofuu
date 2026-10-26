// t77_net_policy.js — R7.1/R7.2 (net dimension): wildcard host patterns match
// only the intended suffix, and `:port`, `user@`, and `[v6]` normalize before
// the match — the same decision on the JS and CLI surfaces.
//
// Run: ./sofuu run scripts/contract/harness/t77_net_policy.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);

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
    const c1 = await sh("./sofuu policy why net.fetch host=sub.evil.example:8443 --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ net: { deny: ["*.evil.example"] } }) + "\nEOP");
    const c1j = P(String(c1.out).trim().split("\n").pop());
    assert("CLI: the identical deny + rule",
        c1j.decision === "deny" && String(c1j.rule).indexOf("net") >= 0);
    const c2 = await sh("./sofuu policy why net.fetch host=good.example.org --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ net: { deny: ["*.evil.example"] } }) + "\nEOP");
    const c2j = P(String(c2.out).trim().split("\n").pop());
    assert("CLI: the identical allow",
        c2j.decision === "allow");

    sofuu.policy.set("{}");
    console.log(`\n=== t77 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t77:", (e && e.message) || e); process.exit(1); });
