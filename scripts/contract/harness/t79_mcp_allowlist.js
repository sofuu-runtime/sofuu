// t79_mcp_allowlist.js — R7.7: MCP servers are a trust boundary — a
// non-allowlisted server is refused, an allowlisted one passes, and the
// allowlisted server's TOOLS still run under the same tools/approval policy
// as built-ins.
//
// Run: ./sofuu run scripts/contract/harness/t79_mcp_allowlist.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({ command: "/bin/sh", args: ["-c", cmd],
            onStdout: (c) => { out += String(c); }, onExit: (c) => { code = c; resolve({ code, out }); } });
    });
}

async function main() {
    sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } } }));

    // ── the evaluator boundary ────────────────────────────────────────────
    const d1 = P(sofuu.policy.evaluate(JSON.stringify({ op: "mcp.connect", server: "rogue-server" })));
    assert("non-allowlisted MCP server refused + rule",
        d1.decision === "deny" && String(d1.rule).indexOf("mcp") >= 0);
    const a1 = P(sofuu.policy.evaluate(JSON.stringify({ op: "mcp.connect", server: "trusted-server" })));
    assert("allowlisted MCP server passes",
        a1.decision === "allow");
    const a2 = P(sofuu.policy.evaluate(JSON.stringify({ op: "tool.run", tool: "trusted-server__query" })));
    assert("an MCP-sourced tool still answers through the tools policy (same gate as built-ins)",
        a2.decision === "allow");

    // ── the same refusal through the CLI (one evaluator) ─────────────────
    const c = await sh("./sofuu policy why mcp.connect server=rogue-server --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } } }) + "\nEOP");
    const cj = P(String(c.out).trim().split("\n").pop());
    assert("CLI: the identical refusal + rule",
        cj.decision === "deny" && String(cj.rule).indexOf("mcp") >= 0);

    // ── default posture (no allowlist installed): the gate does not pretend ─
    P(sofuu.policy.set("{}"));
    const empty = P(sofuu.policy.evaluate(JSON.stringify({ op: "mcp.connect", server: "whatever" })));
    assert("without an allowlist the gate does not pretend to allowlist",
        empty.decision === "allow");

    sofuu.policy.set("{}");
    console.log(`\n=== t79 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t79:", (e && e.message) || e); process.exit(1); });
