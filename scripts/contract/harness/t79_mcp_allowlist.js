// t79_mcp_allowlist.js — R7.7: MCP servers are a trust boundary — a
// non-allowlisted server is refused BEFORE it is started (the refusal
// reaches no backend at all), an allowlisted one connects for real, and
// the allowlisted server's TOOLS still run under the same tools policy as
// the built-ins.
//
// The evaluator checks are pure policy decisions: no backend is involved.
// The server checks use the REAL local MCP file server — it reads an actual
// temporary file and writes its own receipt only after that read succeeded,
// so "the call never reached the server" is observable on disk.
//
// RESOURCE: real: the local MCP file server (the allowlist gate is evaluated in process)
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

/* The MCP server script ships next to this harness; the binary that runs it
 * is the one executing this file, so the server is the same build. */
function siblingScript(name) {
    const argv = (typeof process !== "undefined" && process.argv) || [];
    const self = String(argv[1] || "");
    const candidates = [];
    if (self) candidates.push(self.replace(/[^/]*$/, name));
    candidates.push("scripts/contract/harness/" + name);
    for (let i = 0; i < candidates.length; i++) {
        if (candidates[i] && sofuu.fs.existsSync(candidates[i])) return candidates[i];
    }
    return "";
}

const BINARY = String((typeof process !== "undefined" && process.argv && process.argv[0]) || "./sofuu");

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

    // ── the REAL server: the allowlist is enforced before dispatch ───────
    const serverScript = siblingScript("real_file_mcp_server.js");
    if (!serverScript) {
        console.error("  ❌ FAIL: the real MCP file server script was not found next to this harness");
        results.failed++;
    } else {
        const dir = "/tmp/t79-mcp-" + (Date.now() % 1000000);
        const sample = dir + "/sample.txt";
        const receipt = dir + "/receipt.json";
        const starts = dir + "/starts.log";
        const marker = "T79-MARKER-4b81e0c7";
        await sofuu.fs.mkdir(dir, { recursive: true });
        await sofuu.fs.writeFile(sample, marker);
        const spec = { command: BINARY, args: ["run", serverScript, sample, receipt, starts] };

        // 1. A server that is not allowlisted is refused, and the refusal
        //    happens BEFORE the process is spawned: the server's own startup
        //    log never appears.
        sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } } }));
        let refused = null;
        try { await sofuu.mcp.connect(spec, { server: "rogue-server" }); }
        catch (e) { refused = e; }
        assert("real server: a non-allowlisted server is refused",
            !!refused && String(refused.code) === "E_POLICY_DENIED");
        assert("real server: the refused connect never started the server (no backend reached)",
            !sofuu.fs.existsSync(starts) && !sofuu.fs.existsSync(receipt));

        // 2. The allowlisted server connects for real and advertises its tools.
        sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } } }));
        let client = null;
        try { client = await sofuu.mcp.connect(spec, { server: "trusted-server" }); }
        catch (e) { console.error("  ❌ FAIL: the allowlisted server did not connect: " + String(e && e.message || e)); results.failed++; }
        if (client) {
            try {
                const listing = await client.listTools();
                assert("real server: the allowlisted server advertises its real tools",
                    !!listing && Array.isArray(listing.tools) &&
                    listing.tools.some((t) => t.name === "read_sample"));

                // 3. The allowlisted server's TOOLS still run under the tools
                //    policy: a denied tool is refused and the server never
                //    receives it — its receipt stays absent.
                sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } },
                    tools: { deny: ["read_sample"] } }));
                let toolDenied = null;
                try { await client.call("tools/call", { name: "read_sample", arguments: {} }); }
                catch (e) { toolDenied = e; }
                assert("real server: a policy-denied MCP tool is refused with the typed error",
                    !!toolDenied && String(toolDenied.code) === "E_POLICY_DENIED");
                assert("real server: the denied tool never reached the server (no receipt)",
                    !sofuu.fs.existsSync(receipt));

                // 4. With the deny lifted the same tool really runs: the
                //    server reads the actual file and records it.
                sofuu.policy.set(JSON.stringify({ mcp: { servers: { allow: ["trusted-server"] } } }));
                const result = await client.call("tools/call", { name: "read_sample", arguments: {} });
                const mapped = sofuu.tools.mcpResult(result);
                assert("real server: the allowlisted tool really ran and returned the file",
                    mapped.ok && mapped.text.indexOf(marker) >= 0);
                let recorded = null;
                try { recorded = P(String(await sofuu.fs.readFile(receipt, "utf8"))); } catch (e) {}
                assert("real server: the server's own receipt matches the file it read",
                    !!recorded && recorded.operation === "read_sample" && recorded.contents === marker);
            } finally {
                client.disconnect();
            }
        }

        try { await sofuu.fs.rm(dir); } catch (e) { /* the temp dir is disposable */ }
    }

    sofuu.policy.set("{}");
    console.log(`\n=== t79 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t79:", (e && e.message) || e); process.exit(1); });
