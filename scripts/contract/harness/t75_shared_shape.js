// t75_shared_shape.js — R7.6/R7.1: the shell kill switch and the fs jail
// answer identically on every surface — the same rule ids on JS and CLI, the
// same strip on the HTTP inventory, and no bypass by changing entry points.
//
// Every refusal is checked for its SIDE EFFECT as well: a denied command
// must not run, so the marker file it would have created never appears.
// "Refused" therefore means "never reached the tool", not "the tool answered
// with an error".
//
// RESOURCE: real: the real built-in tools and a real serve process (no model needed)
//
// Run: ./sofuu run scripts/contract/harness/t75_shared_shape.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 120 s"); process.exit(1); }, 120000);

const P = (s) => JSON.parse(s);

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({ command: "/bin/sh", args: ["-c", cmd],
            onStdout: (c) => { out += String(c); }, onExit: (c) => { code = c; resolve({ code, out }); } });
    });
}

function curl(args) {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({ command: "/usr/bin/curl",
            args: ["-s", "-m", "10"].concat(args),
            onStdout: (c) => { out += String(c); }, onExit: (code) => resolve({ code: code, out: out }) });
    });
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/* The binary under test — the one running this harness. Every surface must be
 * the same build, or "the same answer everywhere" means nothing. */
const BIN = String((typeof process !== "undefined" && process.argv && process.argv[0]) || "./sofuu");

async function main() {
    /* The markers a refused command would have created. Their absence is the
     * proof that the refusal happened BEFORE the tool ran. */
    const stamp = Date.now() % 1000000;
    const bashMarker = "/tmp/t75-bash-marker-" + stamp;
    /* A relative path stays inside the project jail, so the fs.write POLICY
     * is what refuses it (an absolute /tmp path would be refused by the jail
     * first and the rule id would be a different one). */
    const fsMarker = "t75-fs-marker-" + stamp + ".txt";
    const escapeMarker = "/tmp/t75-escape-marker-" + stamp + ".txt";

    // ── JS: the evaluator route refuses with the rule ─────────────────────
    sofuu.policy.set(JSON.stringify({ shell: { enabled: false }, fs: { write: "off" } }));
    let polErr = "";
    try { await sofuu.tools.TOOLS.bash.execute({ command: "touch " + bashMarker }); }
    catch (e) { polErr = String(e && e.message || e); }
    assert("JS: policy shell.enabled=false refuses the real tool citing the rule",
        polErr.indexOf("denied by policy") >= 0 && polErr.indexOf("R7.6/shell.enabled") >= 0);
    assert("JS: the refused shell command never ran (no marker file)",
        !sofuu.fs.existsSync(bashMarker));
    let fsPolErr = "";
    try { await sofuu.tools.TOOLS.write_file.execute({ path: fsMarker, content: "x" }); }
    catch (e) { fsPolErr = String(e && e.message || e); }
    assert("JS: policy fs.write=off refuses the real tool citing the rule",
        fsPolErr.indexOf("R7.6/fs.write") >= 0);
    assert("JS: the refused write never happened (no file)",
        !sofuu.fs.existsSync(fsMarker));
    sofuu.policy.set("{}");

    // ── CLI: the same refusal, same rule, through one evaluator ───────────
    const c1 = await sh(BIN + " policy why shell.run command='echo hi' --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ shell: { enabled: false } }) + "\nEOP");
    const c1j = P(String(c1.out).trim().split("\n").pop());
    assert("CLI: shell refusal + the SAME rule id",
        c1j.decision === "deny" && c1j.rule === "R7.6/shell.enabled");
    const c2 = await sh(BIN + " policy why fs.write path=t75-w.txt --policy /dev/stdin <<'EOP'\n" +
        JSON.stringify({ fs: { write: "off" } }) + "\nEOP");
    const c2j = P(String(c2.out).trim().split("\n").pop());
    assert("CLI: fs-write refusal + the SAME rule id",
        c2j.decision === "deny" && c2j.rule === "R7.6/fs.write");

    // ── CLI: SOFUU_NO_SHELL (the R7.6 kill switch) strips the belt ────────
    const nos = await sh("SOFUU_NO_SHELL=1 ./sofuu eval 'console.log(sofuu.tools.GROUP.indexOf(\"bash\"))'");
    assert("CLI: SOFUU_NO_SHELL strips bash from the tool group",
        String(nos.out).trim().endsWith("-1"));

    // ── HTTP: SOFUU_NO_SHELL=1 on the served surface — same strip, no bypass
    // The server is spawned DIRECTLY (no shell wrapper to orphan it) and
    // needs no model: the tool inventory does not depend on one, and the
    // synthetic contract fixture is a parser fixture, never a served model.
    const port = 38900 + (Date.now() % 600);
    let svr = null, up = false;
    try {
        const pwd = (await sh("pwd")).out.trim();
        svr = sofuu.spawn({
            command: pwd + "/sofuu",
            args: ["serve", "--port", String(port), "--host", "127.0.0.1", "--token", "t75-tok"],
            env: { SOFUU_NO_SHELL: "1" },
            onStdout: () => {}, onStderr: () => {}, onExit: () => {},
        });
        for (let i = 0; i < 60 && !up; i++) {
            await wait(250);
            const probe = await curl(["-H", "Authorization: Bearer t75-tok", "http://127.0.0.1:" + port + "/v1/tools"]);
            up = String(probe.out).indexOf('"data"') >= 0;
        }
        assert("HTTP: serve came up under SOFUU_NO_SHELL=1", up);
        if (up) {
            const tools = await curl(["-H", "Authorization: Bearer t75-tok", "http://127.0.0.1:" + port + "/v1/tools"]);
            const names = /^[0-9]+$/.test(String(tools.out).trim()) ? [] : (String(tools.out).match(/"name":"([a-z_]+)"/g) || []);
            const containsBash = names.join(",").indexOf("bash") >= 0;
            assert("HTTP: bash is absent from the served tool inventory", up && !containsBash);
            assert("HTTP: the inventory still lists the other tools (the strip removed bash, not the tools)",
                names.join(",").indexOf("read_file") >= 0);
            const httpMarker = "/tmp/t75-http-marker-" + stamp;
            const callBash = await curl(["-X", "POST", "-H", "Authorization: Bearer t75-tok",
                "-H", "Content-Type: application/json",
                "-d", JSON.stringify({ arguments: { command: "touch " + httpMarker } }),
                "http://127.0.0.1:" + port + "/v1/tools/bash/call"]);
            let b = null;
            try { b = P(String(callBash.out)); } catch (e) {}
            assert("HTTP: a direct bash call is refused, never executed (R7.2)",
                !(b && b.ok === true) && !!(b && b.error));
            assert("HTTP: the refused command never ran (no marker file)",
                !sofuu.fs.existsSync(httpMarker));
        }
    } finally {
        try { svr && svr.kill && svr.kill(); } catch (e) {}
        await wait(300);
    }

    // ── FS jail: the lexical escape refusal is the same error on JS & CLI ─
    const escJs = await sh(BIN + " eval 'sofuu.tools.TOOLS.write_file.execute({path:\"" + escapeMarker + "\", content:\"x\"}).then((r)=>console.log(String(r).slice(0,60)), (e)=>console.log(e.message))' 2>&1 | tail -1");
    assert("FS jail: the escape refusal names the jail (JS surface = CLI binary)",
        String(escJs.out).indexOf("escapes the project directory") >= 0 &&
        String(escJs.out).indexOf("jailed") >= 0);
    assert("FS jail: the escaped write never happened (no file)",
        !sofuu.fs.existsSync(escapeMarker));

    console.log(`\n=== t75 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t75:", (e && e.message) || e); process.exit(1); });
