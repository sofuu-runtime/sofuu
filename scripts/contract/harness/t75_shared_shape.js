// t75_shared_shape.js — R7.6/R7.1: the shell kill switch and the fs jail
// answer identically on every surface — the same rule ids on JS and CLI, the
// same strip on the HTTP inventory, and no bypass by changing entry points.
//
// Run: ./sofuu run scripts/contract/harness/t75_shared_shape.js
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
    // ── JS: the evaluator route refuses with the rule ─────────────────────
    sofuu.policy.set(JSON.stringify({ shell: { enabled: false }, fs: { write: "off" } }));
    let polErr = "";
    try { await sofuu.tools.TOOLS.bash.execute({ command: "echo hi" }); }
    catch (e) { polErr = String(e && e.message || e); }
    assert("JS: policy shell.enabled=false refuses the real tool citing the rule",
        polErr.indexOf("denied by policy") >= 0 && polErr.indexOf("R7.6/shell.enabled") >= 0);
    let fsPolErr = "";
    try { await sofuu.tools.TOOLS.write_file.execute({ path: "t75-w.txt", content: "x" }); }
    catch (e) { fsPolErr = String(e && e.message || e); }
    assert("JS: policy fs.write=off refuses the real tool citing the rule",
        fsPolErr.indexOf("R7.6/fs.write") >= 0);
    sofuu.policy.set("{}");

    // ── CLI: the same refusal, same rule, through one evaluator ───────────
    const c1 = await sh("./sofuu policy why shell.run command='echo hi' --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ shell: { enabled: false } }) + "\nEOP");
    const c1j = P(String(c1.out).trim().split("\n").pop());
    assert("CLI: shell refusal + the SAME rule id",
        c1j.decision === "deny" && c1j.rule === "R7.6/shell.enabled");
    const c2 = await sh("./sofuu policy why fs.write path=t75-w.txt --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ fs: { write: "off" } }) + "\nEOP");
    const c2j = P(String(c2.out).trim().split("\n").pop());
    assert("CLI: fs-write refusal + the SAME rule id",
        c2j.decision === "deny" && c2j.rule === "R7.6/fs.write");

    // ── CLI: SOFUU_NO_SHELL (the R7.6 kill switch) strips the belt ────────
    const nos = await sh("SOFUU_NO_SHELL=1 ./sofuu eval 'console.log(sofuu.tools.GROUP.indexOf(\"bash\"))'");
    assert("CLI: SOFUU_NO_SHELL strips bash from the tool group",
        String(nos.out).trim().endsWith("-1"));

    // ── HTTP: SOFUU_NO_SHELL=1 on the served surface — same strip, no bypass
    const port = 38900 + (Date.now() % 600);
    let svr = null, up = false, bin = "";
    try {
        const pwd = (await sh("pwd")).out.trim();
        bin = pwd + "/sofuu";
        svr = sofuu.spawn({ command: "/bin/sh", args: ["-c",
            "SOFUU_NO_SHELL=1 " + bin + " serve --port " + port + " --host 127.0.0.1 --token t75-tok --model contract/fixtures/sofuu-fixture.gguf"],
            onStdout: () => {}, onStderr: () => {}, onExit: () => {} });
        for (let i = 0; i < 60 && !up; i++) {
            await new Promise((r) => setTimeout(r, 250));
            const probe = await sh("curl -s -m 3 -H 'Authorization: Bearer t75-tok' http://127.0.0.1:" + port + "/v1/models");
            up = String(probe.out).indexOf('"data"') >= 0;
        }
        assert("HTTP: serve came up under SOFUU_NO_SHELL=1", up);
        if (up) {
            const tools = await sh("curl -s -m 5 -H 'Authorization: Bearer t75-tok' http://127.0.0.1:" + port + "/v1/tools");
            const names = /^[0-9]+$/.test(String(tools.out).trim()) ? [] : (String(tools.out).match(/"name":"([a-z_]+)"/g) || []);
            const containsBash = names.join(",").indexOf("bash") >= 0;
            assert("HTTP: bash is absent from the served tool inventory", up && !containsBash);
            const callBash = await sh("curl -s -m 5 -X POST -H 'Authorization: Bearer t75-tok' -H 'Content-Type: application/json' -d '{\"arguments\":{\"command\":\"echo hi\"}}' http://127.0.0.1:" + port + "/v1/tools/bash/call");
            const b = P(String(callBash.out));
            assert("HTTP: a direct bash call is refused, never executed (R7.2)",
                !(b && b.ok === true));
        }
    } finally {
        try { svr && svr.kill && svr.kill(); } catch (e) {}
        await new Promise((r) => setTimeout(r, 300));
    }

    // ── FS jail: the lexical escape refusal is the same error on JS & CLI ─
    const escJs = await sh("./sofuu eval 'sofuu.tools.TOOLS.write_file.execute({path:\"/tmp/t75-escape.txt\", content:\"x\"}).then((r)=>console.log(String(r).slice(0,60)), (e)=>console.log(e.message))' 2>&1 | tail -1");
    assert("FS jail: the escape refusal names the jail (JS surface = CLI binary)",
        String(escJs.out).indexOf("escapes the project directory") >= 0 &&
        String(escJs.out).indexOf("jailed") >= 0);

    console.log(`\n=== t75 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t75:", (e && e.message) || e); process.exit(1); });
