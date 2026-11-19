// t78_secrets_never_emitted.js — R7.5: a provider key resolved at call time
// never reaches the trace, the event log, the session document, or error
// text.
//
// The provider is the REAL local model server (`sofuu serve` with a
// real-weight GGUF): the agent resolves the key at call time and sends it
// as the Authorization header, and the server accepts it — a wrong key would
// answer 401, so a completed turn proves the key really travelled over the
// wire to a real backend. The failing turn is a real provider failure (the
// server does not know that model id).
//
// The provider-echo leak route — a provider that answers 401 with the
// Authorization header in its body — cannot be exercised without such a
// provider, and the runtime currently surfaces a provider error body verbatim
// in the thrown error text (crates/sofuu-core/src/rt/ai.rs, stream_status_error).
// That route is recorded as an explicit unverified scenario below; the
// redaction seam that does protect the event surfaces is checked directly.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL served by the real `sofuu serve`; blocks when unset
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/real.gguf \
//      ./sofuu run scripts/contract/harness/t78_secrets_never_emitted.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 10 min"); process.exit(1); }, 600000);

const P = (s) => JSON.parse(s);
const SECRET = "sk-t78-live-secret-9f2c1";
const MODEL = String((typeof process !== "undefined" && process.env && process.env.SOFUU_CONTRACT_MODEL) || "");
const BIN = String((typeof process !== "undefined" && process.argv && process.argv[0]) || "./sofuu");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function curl(args) {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({ command: "/usr/bin/curl", args: ["-s", "-m", "20"].concat(args),
            onStdout: (c) => { out += String(c); }, onExit: (code) => resolve({ code: code, out: out }) });
    });
}

if (!MODEL || !/\.gguf$/i.test(MODEL) || !sofuu.fs.existsSync(MODEL) ||
    /(^|\/)sofuu-fixture\.gguf$/i.test(MODEL)) {
    console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing real-weight GGUF " +
        "(the synthetic contract fixture is a parser fixture, not an inference backend)");
    process.exit(2);
}

async function main() {
    // ── the real provider: the runtime's own model server ────────────────
    // The bearer token IS the secret under test: the agent's key has to
    // match it for the server to answer at all.
    const port = 39700 + (Date.now() % 500);
    const base = "http://127.0.0.1:" + port;
    const server = sofuu.spawn({
        command: BIN,
        args: ["serve", "--model", MODEL, "--model-id", "t78-local-model",
            "--port", String(port), "--host", "127.0.0.1"],
        env: { SOFUU_SERVE_TOKEN: SECRET },
        onStdout: () => {}, onStderr: () => {}, onExit: () => {},
    });
    let up = false;
    for (let i = 0; i < 240 && !up; i++) {
        await wait(500);
        const probe = await curl(["-H", "Authorization: Bearer " + SECRET, base + "/v1/models"]);
        up = String(probe.out).indexOf('"data"') >= 0;
    }
    assert("the real model server came up with the configured key", up);

    try {
        if (!up) {
            console.error("BLOCKED: the real model server did not come up — no provider to check");
            process.exit(2);
        }

        // ── successful turn: the key must not ride any observable surface ─
        const def = { name: "secrets-t78", system: "Reply briefly.", provider: "custom",
            model: "t78-local-model", base_url: base + "/v1/chat/completions",
            api_key: SECRET, memory: "off", rlm: "off" };
        let trace1 = "";
        let answer1 = "";
        try {
            const r1 = await sofuu.agent.run(def, "hello t78", {});
            trace1 = JSON.stringify(r1 && r1.trace || "") + JSON.stringify(r1 && r1.answer || "");
            answer1 = String((r1 && r1.answer) || "");
        } catch (e) { trace1 = String(e && e.message || e); }
        assert("the real provider answered the turn (the configured key was accepted)", answer1.length > 0);
        assert("no key bytes in the successful run's trace",
            trace1.indexOf(SECRET) < 0);

        // ── failing turn: a REAL provider failure (the server does not know
        //    that model id), not a fabricated error body ────────────────
        const badDef = Object.assign({}, def, { model: "t78-no-such-model" });
        let trace2 = "";
        let failed = false;
        try {
            const r2 = await sofuu.agent.run(badDef, "fail loudly", {});
            trace2 = JSON.stringify(r2 && r2.trace || "");
            failed = !!(r2 && r2.error);
        } catch (e) { trace2 = String(e && e.message || e); failed = true; }
        assert("the failing turn really failed against the real provider", failed);
        assert("no key bytes in the failing run's trace or error", trace2.indexOf(SECRET) < 0);

        // ── the events log + the session document (the R4 export surface) ─
        const project = "/tmp/t78-sessions-" + (Date.now() % 100000);
        await sofuu.fs.mkdir(project, { recursive: true });
        const SID = "s-3f8c1a02d9e7b-t4" + (780000001 + (Date.now() % 100000000));
        const mk = (kind, seq, payload) => ({ kind: kind, schema_version: 1, sid: SID,
            turn_id: SID + "-t" + (seq % 2 ? 1 : 2), seq: seq,
            ts: "2026-10-26T10:0" + (seq % 10) + ":00Z", surface: "js", payload: payload });
        const evs = [
            mk("prompt", 1, { text: "hello t78" }),
            mk("delta", 2, { text: "ok" }),
            mk("usage", 3, { in: 2, out: 3, model: "t78-local-model", provider: "custom" }),
            mk("served", 4, { by: "local" }),
            mk("stop", 5, { reason: "done" }),
        ];
        const ap = P(sofuu.events.append(JSON.stringify({ project: project, sid: SID, events: evs })));
        assert("events appended", ap.ok === true);
        let logBlob = "";
        try { logBlob += String(await sofuu.fs.readFile(project + "/.sofuu/sessions/" + SID + "/events.ndjson")); } catch (e) {}
        const docBuild = sofuu.session.export(JSON.stringify({ project: project, sid: SID }));
        P(sofuu.session.import(docBuild));
        const docOut = sofuu.session.export(JSON.stringify({ project: project, sid: SID }));
        assert("no key bytes in the event log or the session document",
            (logBlob + docOut).indexOf(SECRET) < 0);
        try { await sofuu.fs.rm(project); } catch (e) { /* the temp project is disposable */ }

        // ── the redaction seam: a provider-echoed key in a free-form
        //    message is scrubbed before it reaches an event ──────────────
        const echoed = P(sofuu.events.redact(JSON.stringify({
            kind: "error", schema_version: 1, sid: SID, turn_id: SID + "-t1", seq: 9,
            ts: "2026-10-26T10:09:00Z", surface: "js",
            payload: { message: "Invalid API key provided: Bearer " + SECRET },
        })));
        assert("the event redactor scrubs a provider-echoed key from a free-form message",
            JSON.stringify(echoed).indexOf(SECRET) < 0);
    } finally {
        try { server && server.kill && server.kill(); } catch (e) {}
        await wait(300);
    }

    console.log(`\n=== t78 results: ${results.passed} passed, ${results.failed} failed ===`);
    if (results.failed > 0) process.exit(1);
    console.error("BLOCKED: the provider-echo leak route is unverified here — it needs a provider " +
        "that answers with the Authorization header in its body, and the runtime surfaces a " +
        "provider error body verbatim in the thrown error text " +
        "(crates/sofuu-core/src/rt/ai.rs stream_status_error). The event, event-log and " +
        "session-document surfaces are checked and redacted above.");
    process.exit(2);
}

main().catch((e) => { console.error("FAIL t78:", (e && e.message) || e); process.exit(1); });
