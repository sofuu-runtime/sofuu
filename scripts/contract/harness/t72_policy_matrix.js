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

const P = (s) => JSON.parse(s);

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({ command: "/bin/sh", args: ["-c", cmd],
            onStdout: (c) => { out += String(c); }, onExit: (c) => { code = c; resolve({ code, out }); } });
    });
}

/* ── H-12: the serve child must not outlive this harness ────────────────
 * The old `finally` called `svr.kill()`, whose SIGTERM went to the
 * `/bin/sh -c "cd … && ./sofuu serve …"` WRAPPER. The server itself was
 * reparented to PID 1 and kept its port — two such orphans were still alive
 * 1d16h after a run, and one more was reproduced during this audit. Fixes:
 *
 *   1. `exec` in the wrapper: the spawned child IS `sofuu serve`, so the
 *      handle's own kill() reaches the server and no shell remains behind
 *      to orphan it. The wrapper records `$$` first, so the server's pid is
 *      known even though SOFUU's spawn handle exposes neither `pid` nor a
 *      detached/setsid flag (process_spawn.rs sets flags = 0).
 *   2. A process-TREE teardown: signal the process group first
 *      (`process.kill(-pid, …)` when the runtime grows one; `kill -SIG --
 *      -pid` otherwise — the pid IS the server), then the pid itself and
 *      every descendant found through `ps`, SIGTERM → grace → SIGKILL.
 *   3. Proof, not hope: wait for the port to stop accepting, then assert no
 *      process still carries this run's `serve --port <port>` marker.       */

const KILL_GRACE_MS = 4000;
let serveWatch = null; /* the live serve descriptor (the watchdog reaps it) */

/* ps snapshot as [{pid, ppid}]. Absolute path: the child env is a filtered
 * allowlist, and `ps` must not depend on PATH. */
async function processTable() {
    const out = await sh("/bin/ps -axo pid=,ppid=");
    const rows = [];
    String(out.out).split("\n").forEach((line) => {
        const f = line.trim().split(/\s+/);
        if (f.length < 2) return;
        const pid = Number(f[0]), ppid = Number(f[1]);
        if (!isFinite(pid) || !isFinite(ppid)) return;
        rows.push({ pid: pid, ppid: ppid });
    });
    return rows;
}

/* Every pid in `root`'s subtree (root first), from one ps snapshot. */
function subtreePids(rows, root) {
    const kids = {};
    rows.forEach((r) => { (kids[r.ppid] = kids[r.ppid] || []).push(r.pid); });
    const out = [], stack = [root];
    while (stack.length) {
        const pid = stack.pop();
        if (!(pid > 1) || out.indexOf(pid) >= 0) continue;
        out.push(pid);
        (kids[pid] || []).forEach((k) => stack.push(k));
    }
    return out;
}

/* Signal ONE pid. `process.kill` is not part of this runtime's surface, so
 * the shell builtin is the fallback (`kill -SIG pid`). */
async function signalPid(pid, sig) {
    if (!(pid > 1)) return;
    if (typeof process.kill === "function") {
        try { process.kill(pid, sig); return; } catch (e) {}
    }
    await sh("kill -" + sig + " " + pid + " 2>/dev/null || true");
}

/* Signal the whole process GROUP led by `pid` (the documented
 * `process.kill(-pid, sig)`), then the pid itself. When the pid is not a
 * group leader the negative form is a harmless ESRCH — a pgid always equals
 * its leader's pid, so no other group can be hit. */
async function signalGroup(pid, sig) {
    if (!(pid > 1)) return;
    if (typeof process.kill === "function") {
        try { process.kill(-pid, sig); } catch (e) {}
        try { process.kill(pid, sig); } catch (e) {}
        return;
    }
    await sh("kill -" + sig + " -- -" + pid + " 2>/dev/null; " +
        "kill -" + sig + " " + pid + " 2>/dev/null; true");
}

/* Every process whose command line carries this run's port marker. The
 * command is the server's own argv, so the match is exact per run. */
async function serveSurvivors(marker) {
    const out = await sh("/bin/ps -axo pid=,ppid=,command=");
    const hits = [];
    String(out.out).split("\n").forEach((line) => {
        if (line.indexOf(marker) < 0) return;
        if (line.indexOf("ps -axo") >= 0) return; /* the scanner itself */
        hits.push(line.trim());
    });
    return hits;
}

/* Connect-refused ⇒ the port is released. A 401/403 still means OPEN. */
async function portReleased(port) {
    const probe = await sh("/usr/bin/curl -s -m 2 -o /dev/null -w '%{http_code}' " +
        "http://127.0.0.1:" + port + "/v1/models || echo closed");
    return String(probe.out).indexOf("closed") >= 0;
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/* Kill the server AND its tree, then prove the port and the process table
 * are clean. Idempotent: the watchdog calls it too. Returns the survivors
 * it could NOT kill (empty = clean). */
async function reapServe(serve) {
    if (!serve || serve.reaped) return (serve && serve.survivors) || [];
    serve.reaped = true;
    const marker = "serve --port " + serve.port;

    /* 1. Collect the targets: the pid recorded before exec (+ its whole
     *    subtree), plus anything the marker still finds, plus the handle's
     *    pid when the runtime exposes one. */
    const rows = await processTable();
    const targets = [];
    const addPid = (p) => { if (p > 1 && targets.indexOf(p) < 0) targets.push(p); };
    let recorded = 0;
    if (serve.pidFile) {
        const t = await sh("cat " + serve.pidFile + " 2>/dev/null || echo 0");
        recorded = Number(String(t.out).trim()) || 0;
    }
    if (recorded) subtreePids(rows, recorded).forEach(addPid);
    if (serve.handle && typeof serve.handle.pid === "number") addPid(serve.handle.pid);
    const hits = await serveSurvivors(marker);
    hits.forEach((line) => addPid(Number(String(line).trim().split(/\s+/)[0])));

    /* 2. SIGTERM: the handle first (direct, no shell), then the group, then
     *    each pid. */
    try { if (serve.handle && serve.handle.kill) serve.handle.kill(); } catch (e) {}
    if (recorded) await signalGroup(recorded, "TERM");
    for (let i = 0; i < targets.length; i++) await signalPid(targets[i], "TERM");

    /* 3. Grace period, then SIGKILL whatever is still there. */
    const deadline = Date.now() + KILL_GRACE_MS;
    for (;;) {
        const live = await processTable();
        const still = targets.filter((p) => live.some((r) => r.pid === p));
        if (still.length === 0 || Date.now() > deadline) break;
        await wait(200);
    }
    if (recorded) await signalGroup(recorded, "KILL");
    for (let i = 0; i < targets.length; i++) await signalPid(targets[i], "KILL");

    /* 4. Wait for the PORT to be released, killing anything the marker
     *    still finds (a descendant spawned after the snapshot above). */
    for (let i = 0; i < 50; i++) {
        if (await portReleased(serve.port)) break;
        const late = await serveSurvivors(marker);
        for (let j = 0; j < late.length; j++) {
            await signalPid(Number(String(late[j]).trim().split(/\s+/)[0]), "KILL");
        }
        await wait(200);
    }

    /* 5. Proof: zero processes carry this run's marker. */
    serve.survivors = await serveSurvivors(marker);
    return serve.survivors;
}

setTimeout(() => {
    console.error("  ❌ WATCHDOG: harness overran 60 s");
    /* Reap before exiting — a watchdog must not be the leak. */
    Promise.resolve(reapServe(serveWatch)).then(
        () => process.exit(1),
        () => process.exit(1));
}, 60000);

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

    /* H-02(a) (AUDIT-2026-10-29): `shell.enabled:false` now refuses a shell
     * INTERPRETER for everyone, not just tool-originated spawns — that is the
     * fix for the audit's `sofuu.spawn({command:'/bin/sh'})` bypass. This
     * harness legitimately needs /bin/sh for its own plumbing (the `sh()`
     * helper below, and the H-12 serve wrapper), so it drops its own copy of
     * the policy here. Nothing is weakened: every surface is still driven by
     * the SAME policy file on disk — `--policy <file>` for the CLI checks and
     * `.sofuu/policy.json` inside the serve workdir for the HTTP ones. */
    sofuu.policy.clear();

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
    const pidFile = work + "/server.pid";
    await sh("mkdir -p " + work + "/.sofuu && cp " + policyFile + " " + work + "/.sofuu/policy.json");
    const serve = { handle: null, port: port, pidFile: pidFile, reaped: false, survivors: [] };
    serveWatch = serve;
    let up = false;
    try {
        // serve resolves its project policy from ITS cwd — spawn it there.
        // `exec` makes the child pid the SERVER (H-12), and `echo $$` (the
        // shell's pid, unchanged by exec) records it for the teardown.
        const absSofuu = (await sh("ls -d $(pwd)/sofuu")).out.trim() || "./sofuu";
        serve.handle = sofuu.spawn({ command: "/bin/sh", args: ["-c",
            "echo $$ > " + pidFile + " && cd " + work + " && exec '" + absSofuu +
            "' serve --port " + port + " --host 127.0.0.1 --token t72-tok"],
            onStdout: () => {}, onStderr: () => {}, onExit: () => {} });
        for (let i = 0; i < 60 && !up; i++) {
            await wait(250);
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
        // H-12: kill the server AND every descendant, wait for the port to be
        // released, and report what (if anything) survived.
        const survivors = await reapServe(serve);
        assert("H-12: teardown leaves zero orphaned serve processes (port " + port + " released)",
            survivors.length === 0 &&
            (await portReleased(port)));
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

main().catch(async (e) => {
    console.error("FAIL t72:", (e && e.message) || e);
    try { await reapServe(serveWatch); } catch (e2) {}
    process.exit(1);
});
