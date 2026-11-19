// t23_cancel_tool.js — R2.3: a cancelled run kills the in-flight tool's
// subprocess — no orphan, prompt settle.
//
// The backend is REAL on every leg: the agent's model is an existing local
// GGUF loaded by the in-process inference engine (provider "local"), the tool
// is the REAL built-in bash tool, and the tool's subprocess is a REAL child
// process. The cancel is the REAL sofuu.agent.cancel(signal) path, whose
// tool-kill abort terminates the live child through sofuu.tools.killActive.
// The old scripted provider (which fabricated the tool_call frame) is gone.
//
// A small local model decides whether to call the tool, so the case retries
// a bounded number of times and reports the scenario as BLOCKED — never as a
// pass — when the model never issues the command.
//
// Requires SOFUU_CONTRACT_MODEL to name an existing real-weight GGUF (a
// tool-capable local model is preferred; the case reports BLOCKED when the
// selected model never issues the command). A missing model is a blocked
// prerequisite.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL plus the real bash tool's own subprocess; blocks when unset or when the model never calls the tool
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      ./sofuu run scripts/contract/harness/t23_cancel_tool.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 420 s"); process.exit(1); }, 420000);

const SIGNAL = "t23-cancel";
const MAX_ATTEMPTS = 3;
const ATTEMPT_MS = 120000;

// A unique sleep DURATION is portable (it stays the marked process the
// direct child — dash execs a single simple command in place), keeps the
// marked process out of unrelated wait loops, and no other process on the
// box plausibly sleeps for exactly this many seconds.
const SLEEP_SECONDS = 30000 + (Date.now() % 20000);
const MARKER = "sleep " + SLEEP_SECONDS;

/* The real local weights this case may load. The synthetic contract fixture
 * and the vocabulary-only GGUFs are parser fixtures — they are refused here,
 * they are not inference evidence. */
function resolveModel() {
    const env = String((typeof process !== 'undefined' && process.env) ? (process.env.SOFUU_CONTRACT_MODEL || "") : "").trim();
    if (!env) {
        console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing real-weight GGUF " +
            "(no weights are downloaded; the synthetic contract fixture is a parser fixture)");
        process.exit(2);
    }
    if (!/\.gguf$/i.test(env) || !sofuu.fs.existsSync(env) ||
        /(^|\/)sofuu-fixture\.gguf$/i.test(env) || /ggml-vocab-/i.test(env)) {
        console.error("BLOCKED: SOFUU_CONTRACT_MODEL must name an existing real-weight GGUF " +
            "(not a synthetic or vocabulary fixture): " + env);
        process.exit(2);
    }
    return env;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function pgrepChild() {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "pgrep", args: ["-f", MARKER],
            onStdout: (c) => { out += String(c); },
            onExit: () => resolve(out.trim().split(/\s+/).filter(Boolean)),
        });
    });
}

async function main() {
    const modelPath = resolveModel();
    console.log("  ℹ model: " + modelPath + "  (command: " + MARKER + ")");

    const TASK = "Use the bash tool. Reply with the tool call only. The command is exactly: " + MARKER;
    let cancelled = false;
    let attempts = 0;
    let toolCallSeen = false;

    for (let attempt = 0; attempt < MAX_ATTEMPTS && !cancelled; attempt++) {
        attempts = attempt + 1;
        const agentName = "chaost23_" + attempt;
        sofuu.agent.define({
            name: agentName, system: "You are a helpful assistant.",
            provider: "local", model: modelPath,
            tools: [{ builtin: "bash" }],
            approve: "all", // auto-approve: the harness cancels, not the gate
        });

        let settled = null, settleRejected = null;
        const runP = sofuu.agent.run(agentName, TASK, {
            signal: SIGNAL,
            // approve:'all' still routes through the HOST's callback — fail-closed
            // without one (agent.js refuses to execute on a missing decider).
            // The decider is a RUN opt, not a def field.
            onApproval: function () { return true; },
        });
        // Attach handlers immediately — an unhandled rejection stalls the loop.
        // A rejection after cancel is a legitimate terminal (the contract asks
        // for prompt termination, not success): record it and return, do NOT
        // rethrow into the settle race.
        runP.then(
            (r) => { settled = { how: "resolved", result: r }; },
            (e) => { settleRejected = String((e && e.message) || e); settled = { how: "rejected" }; },
        );

        // Wait until the tool's child is live (or the run ends without one).
        let childLive = false, liveCount = 0;
        const t0 = Date.now();
        while (!childLive && !settled && Date.now() - t0 < ATTEMPT_MS) {
            const live = await pgrepChild();
            if (live.length > 0) { childLive = true; liveCount = live.length; break; }
            await sleep(100);
        }
        if (!childLive) {
            console.error(`  ℹ attempt ${attempts}: the model issued no matching tool call within ${ATTEMPT_MS} ms` +
                (settled ? ` (run ${settled.how})` : "") + " — retrying");
            if (!settled) { try { sofuu.agent.cancel(SIGNAL); } catch (e) { /* nothing to cancel */ } }
            try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (eK) { /* no live child */ }
            await sleep(200);
            continue;
        }
        toolCallSeen = true;
        assert(`the tool's subprocess is live (${liveCount} matching process)`, liveCount > 0);

        const tCancel = Date.now();
        const cRet = sofuu.agent.cancel(SIGNAL);
        assert("cancel accepted via the registered signal", cRet === true);

        // ── the run settles promptly after cancel ──────────────────────
        // Load-aware: the 3 s bound is nominal; a saturated runner delays the
        // harness's own 100 ms ticks, so count that excess and extend the
        // bound the same way T2.1 does. A genuinely stuck run still fails
        // once the slack budget is spent.
        let slack = 0;
        let prev = tCancel;
        while (!settled && Date.now() - tCancel < 3000 + slack) {
            await sleep(100);
            const now = Date.now();
            if (!settled) slack = Math.min(5000, slack + Math.max(0, now - prev - 100));
            prev = now;
        }
        const wall = settled ? Date.now() - tCancel : Date.now() - tCancel;
        const how = settleRejected ? `rejected: ${settleRejected}` : (settled ? settled.how : "timeout");
        assert(
            `run settled after cancel (${wall} ms, ${how}` +
            (slack > 0 ? `, incl. ${slack} ms scheduler slack` : "") + ")",
            settled !== null
        );

        // ── no orphan: the marked child died with the cancel ────────────
        // Reaping is asynchronous (worse on a loaded runner); poll a little
        // longer so the case asserts "no orphan", not "no reap latency".
        let orphans = [];
        const tOrphan = Date.now();
        while (Date.now() - tOrphan < 3000) {
            orphans = await pgrepChild();
            if (orphans.length === 0) break;
            await sleep(100);
        }
        assert("no orphan sleep process after cancel", orphans.length === 0);
        cancelled = true;
    }

    if (!cancelled) {
        if (!toolCallSeen) {
            console.error("BLOCKED: the local model never issued the tool call in " + attempts +
                " attempt(s); the cancel-kills-the-tool-subprocess scenario is unverified on this model");
            process.exit(2);
        }
        console.error("BLOCKED: the tool call did not stay in flight long enough to cancel");
        process.exit(2);
    }

    console.log(`\n=== t23 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t23:", (e && e.message) || e); process.exit(1); });
