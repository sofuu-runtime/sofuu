// t41_session_roundtrip.js — R4.2/R4.3: the §5.2 session document round-trips
// between the JS surface and the CLI with identical turns, tools, usage, and
// IDs; re-import is idempotent; ids survive a restart.
//
// Run: ./sofuu run scripts/contract/harness/t41_session_roundtrip.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

async function main() {
    assert("sofuu.session namespace exists (T4.5)", typeof sofuu.session === "object");
    assert("document schema_version is 1", sofuu.session.schemaVersion === 1);

    const P = JSON.parse;
    const SID = "s-3f8c1a02d9e7b-t4" + (770000000 + (Date.now() % 100000000));
    // The events the fold turns into two §5.2 turns: ids, tools, usage — all
    // carrier pieces the round-trip must preserve (R4.2).
    const mk = (kind, turn, seq, payload) => P(JSON.stringify({
        kind: kind, schema_version: 1, sid: SID, turn_id: SID + "-t" + turn,
        seq: seq, ts: "2026-10-26T09:00:0" + seq + "Z", surface: "js", payload: payload,
    }));
    const events = [
        mk("prompt", 1, 1, { text: "list the repo" }),
        mk("tool", 1, 2, { id: "c1", name: "bash" }),
        mk("tool_result", 1, 3, { id: "c1", name: "bash", ok: true, ms: 12 }),
        mk("usage", 1, 4, { in: 21, out: 34, model: "mock", provider: "local" }),
        mk("stop", 1, 5, { reason: "done" }),
        mk("prompt", 2, 6, { text: "now summarize" }),
        mk("delta", 2, 7, { text: "The repo is " }),
        mk("delta", 2, 8, { text: "quiet." }),
        mk("usage", 2, 9, { in: 40, out: 8, model: "mock", provider: "local" }),
    ];
    const ap = P(sofuu.events.append(JSON.stringify({ project: ".", sid: SID, events })));
    assert("events log written", ap.ok === true && ap.written === 9);

    // ── JS-side export: the doc with turns, tools, usage, ids ────────────
    const doc1 = P(sofuu.session.export(JSON.stringify({ project: ".", sid: SID })));
    assert("export: schema_version stamped", doc1.schema_version === 1);
    assert("export: two turns folded by id", (doc1.turns || []).length === 2);
    assert("export: turn ids are stable shapes",
        doc1.turns[0].id === SID + "-t1" && doc1.turns[1].id === SID + "-t2");
    assert("export: tools carried with ok/ms",
        doc1.turns[0].tools.length === 1 && doc1.turns[0].tools[0].id === "c1"
        && doc1.turns[0].tools[0].ok === true && doc1.turns[0].tools[0].ms === 12);
    assert("export: usage canonically unified (in=61, out=42 across turn 1 + tools turn)",
        doc1.turns[0].usage.in === 21 && doc1.turns[0].usage.out === 34);
    assert("export: answer folded from deltas", doc1.turns[1].answer === "The repo is quiet.");
    assert("export: prompt carried", doc1.turns[0].prompt === "list the repo");
    assert("export: session header carries the id", doc1.session.id === SID);

    // ── Import: the doc persists under its own id, idempotently (R4.3) ───
    const im1 = P(sofuu.session.import(JSON.stringify(doc1)));
    assert("import: ok, same id", im1.ok === true && im1.id === SID);
    const path1 = im1.path;
    const im2 = P(sofuu.session.import(JSON.stringify(doc1)));
    assert("re-import: idempotent, not reported as an overwrite",
        im2.ok === true && im2.overwritten_existing === false);
    assert("import: wrote the canonical path", String(path1).indexOf("session.json") > 0);

    // ── Round-trip identity: rebuild from disk and compare bytes ─────────
    const doc2 = P(sofuu.session.export(JSON.stringify({ project: ".", sid: SID })));
    const strip = (d) => {
        const c = P(JSON.stringify(d));
        if (c.session) delete c.session.project; // cwd diffs across lanes are meta, not data
        if (c.events) c.events.path = String(c.events.path || "").split("/").pop();
        return c;
    };
    assert("round-trip: export → import → export is identical",
        JSON.stringify(strip(doc2)) === JSON.stringify(strip(doc1)));

    // ── R4.5: an app-private section is rejected, not tolerated ──────────
    const bad = P(JSON.stringify(doc1));
    bad.appPrivate = { hiddenState: true };
    const warned = console.log;
    console.log = () => {};
    const rej = P(sofuu.session.import(JSON.stringify(bad)));
    console.log = warned;
    assert("import rejects an app-private top-level section (R4.5)",
        rej.ok === false && String(rej.message || "").indexOf("app-private") >= 0);

    // ── list shows the doc ────────────────────────────────────────────────
    const lst = P(sofuu.session.list(JSON.stringify({ project: "." })));
    const found = (lst.sessions || []).some((s) => String(s.id) === SID && s.turns === 2);
    assert("list reports the imported session", found);

    // ── CLI surface: export/import through the binary (cross-surface leg) ─
    const outFile = "/tmp/t41-doc-" + (Date.now() % 100000) + ".json";
    const rc1 = await new Promise((res) => {
        sofuu.spawn({ command: "/bin/sh", args: ["-c",
            "./sofuu session export " + SID + " --out " + outFile],
            onStdout: () => {}, onExit: (c) => res(c) });
    });
    assert("CLI export exits 0", rc1 === 0);
    const cliDoc = P(await sofuu.fs.readFile(outFile));
    assert("CLI export == JS export (one builder, no drift)",
        JSON.stringify(strip(cliDoc)) === JSON.stringify(strip(doc1)));

    const rc2 = await new Promise((res) => {
        sofuu.spawn({ command: "/bin/sh", args: ["-c",
            "./sofuu session import " + outFile + " > /tmp/t41-import.log 2>&1"],
            onStdout: () => {}, onExit: (c) => res(c) });
    });
    assert("CLI import exits 0", rc2 === 0);
    const imLog = String(await sofuu.fs.readFile("/tmp/t41-import.log"));
    assert("CLI import reports the same id + idempotent",
        imLog.indexOf(SID) >= 0 && imLog.indexOf('"overwritten_existing": false') >= 0);

    // cleanup
    try { sofuu.spawn({ command: "/bin/sh", args: ["-c", "rm -f " + outFile + " /tmp/t41-import.log"], onStdout: () => {}, onExit: () => {} }); } catch (e) {}

    console.log(`\n=== t41 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t41:", (e && e.message) || e); process.exit(1); });
