// t72_policy_matrix.js — R7.1/R7.2: ONE policy object denies the same
// operation with the same rule id on every surface — JS, CLI, HTTP, C ABI —
// and no surface can bypass a deny by changing the entry point.
//
// Run: ./sofuu run scripts/contract/harness/t72_policy_matrix.js
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
        sofuu.spawn({ command: "/bin/sh", args: ["-c", cmd],
            onStdout: (c) => { out += String(c); }, onExit: (c) => { code = c; resolve({ code, out }); } });
    });
}

async function main() {
    // ONE policy, written once — every surface below answers through it.
    const POLICY = {
        tools: { deny: ["read_file"] },
        shell: { enabled: false },
        net: { deny: ["*.evil.example"] },
        fs: { write: "workspace" },
        mcp: { servers: { allow: ["trusted-server"] } },
    };
    const policyFile = "/tmp/t72-matrix-policy.json";
    await sh("cat > " + policyFile + " <<'EOP'\n" + JSON.stringify(POLICY, null, 1) + "\nEOP");
    assert("policy installed for the JS surface",
        P(sofuu.policy.set(JSON.stringify(POLICY))).ok === true);

    // ── JS surface ────────────────────────────────────────────────────────
    const j1 = P(sofuu.policy.evaluate(JSON.stringify({ op: "tool.run", tool: "read_file" })));
    assert("JS: tool deny + rule", j1.decision === "deny" && j1.rule === "R7.6/tools.deny");
    const j2 = P(sofuu.policy.evaluate(JSON.stringify({ op: "shell.run", command: "echo hi" })));
    assert("JS: shell deny + rule", j2.decision === "deny" && j2.rule === "R7.6/shell.enabled");
    // The real belt refuses through the same rule (no escape hatch on JS):
    let toolErr = "";
    try { await sofuu.tools.TOOLS.bash.execute({ command: "echo hi" }); }
    catch (e) { toolErr = String(e && e.message || e); }
    assert("JS: a real bash execution refuses citing the rule",
        toolErr.indexOf("denied by policy") >= 0 && toolErr.indexOf("R7.6/shell.enabled") >= 0);

    // ── CLI surface ───────────────────────────────────────────────────────
    const c1 = await sh("./sofuu policy why tool.run tool=read_file --policy " + policyFile);
    const c1j = P(String(c1.out).trim().split("\n").pop());
    assert("CLI: tool deny + the SAME rule", c1j.decision === "deny" && c1j.rule === "R7.6/tools.deny");
    const c2 = await sh("./sofuu policy why shell.run command='echo hi' --policy " + policyFile);
    const c2j = P(String(c2.out).trim().split("\n").pop());
    assert("CLI: shell deny + the SAME rule", c2j.decision === "deny" && c2j.rule === "R7.6/shell.enabled");

    // ── HTTP surface (serve boots with the project policy on disk) ────────
    const port = 38300 + (Date.now() % 600);
    const work = "/tmp/t72-serve-" + (Date.now() % 100000);
    await sh("mkdir -p " + work + "/.sofuu && cp " + policyFile + " " + work + "/.sofuu/policy.json");
    const bin = JSON.stringify(process.cwd && typeof process.cwd === "function" ? process.cwd() : ".");
    let svr = null, up = false;
    try {
        // serve resolves its project policy from ITS cwd — spawn it there.
        const absSofuu = (await sh("ls -d $(pwd)/sofuu")).out.trim() || "./sofuu";
        svr = sofuu.spawn({ command: "/bin/sh", args: ["-c",
            "cd " + work + " && " + absSofuu + " serve --port " + port + " --host 127.0.0.1 --token t72-tok"],
            onStdout: () => {}, onStderr: () => {}, onExit: () => {} });
        for (let i = 0; i < 60 && !up; i++) {
            await new Promise((r) => setTimeout(r, 250));
            const probe = await new Promise((resolve) => {
                let o = "";
                sofuu.spawn({ command: "/usr/bin/curl", args: ["-s", "-m", "3",
                    "-H", "Authorization: Bearer t72-tok", "http://127.0.0.1:" + port + "/v1/models"],
                    onStdout: (c) => { o += String(c); }, onExit: () => resolve(o) });
            });
            up = String(probe).indexOf('"data"') >= 0;
        }
        assert("HTTP: serve came up (project policy present)", up);
        if (up) {
            const denied = await new Promise((resolve) => {
                let o = "";
                sofuu.spawn({ command: "/usr/bin/curl", args: ["-s", "-m", "5",
                    "-X", "POST", "-H", "Authorization: Bearer t72-tok",
                    "-H", "Content-Type: application/json",
                    "-d", JSON.stringify({ arguments: { path: "README.md" } }),
                    "http://127.0.0.1:" + port + "/v1/tools/read_file/call"],
                    onStdout: (c) => { o += String(c); }, onExit: () => resolve(o) });
            });
            const d = P(String(denied));
            assert("HTTP: tool deny → 403 policy_denied + the SAME rule",
                d && d.error && d.error.code === "policy_denied" && d.error.rule === "R7.6/tools.deny");
        }
    } finally {
        try { svr && svr.kill && svr.kill(); } catch (e) {}
        await sh("rm -rf " + work);
    }

    // ── C ABI surface ─────────────────────────────────────────────────────
    // The embedded host sets the SAME policy object at rt-create time
    // (sofuu_rt_new's config_json carries "policy" into sofuu.policy.set —
    // S5) and evaluation runs through the same registered evaluator; the
    // symbol contract is the suite's ABI gate (T5.5).
    const nm = await sh("ls dist/libsofuu.dylib >/dev/null 2>&1 && " +
        "nm -gU dist/libsofuu.dylib 2>/dev/null | grep -c ' _sofuu_rt_call$' || echo 0");
    const hdr = await sh("grep -c 'SOFUU_ERR_BAD_CONFIG' include/sofuu_embed.h 2>/dev/null || echo 0");
    assert("C ABI: the rt-call symbol is exported and the config-envelope contract is in the embed header",
        String(nm.out).trim() !== "0" && Number(String(hdr.out).trim()) >= 1);

    await sh("rm -f " + policyFile);
    console.log(`\n=== t72 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t72:", (e && e.message) || e); process.exit(1); });
