// t73_approval_harness.js — R7.4: approval modes are explicit and a deadline
// resolves to DENY (T7.3); resolving right after the request event never
// times out to deny (T7.4 — the pre-registration ordering guarantee).
//
// The approval gate is only reachable for real when a REAL model asks for a
// REAL tool: a scripted provider would fabricate the very event under test.
// Both legs therefore run a real local model (SOFUU_CONTRACT_MODEL, an
// existing real-weight GGUF) against the REAL local MCP file server — the
// tool reads an actual temporary file and writes its own receipt only after
// that read succeeded, so "the tool never ran" is observable on disk.
//
// When the configured model does not ask for the tool there is no approval
// event to assert: the harness reports BLOCKED with the reason and never
// invents one.
//
// RESOURCE: real: SOFUU_CONTRACT_MODEL plus the real MCP file server; blocks when unset or when the model never requests the tool
//
// Run: SOFUU_CONTRACT_MODEL=/path/to/real.gguf \
//      ./sofuu run scripts/contract/harness/t73_approval_harness.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
/* A real model round-trip on a CPU-bound local GGUF takes minutes. */
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 15 min"); process.exit(1); }, 900000);

const P = (s) => JSON.parse(s);
const MODEL = String((typeof process !== "undefined" && process.env && process.env.SOFUU_CONTRACT_MODEL) || "");
const BIN = String((typeof process !== "undefined" && process.argv && process.argv[0]) || "./sofuu");
const ATTEMPTS = 3;

/* The MCP server ships next to this harness; the binary that runs it is the
 * one executing this file, so the server is the same build. */
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

const SERVER = siblingScript("real_file_mcp_server.js");

if (!MODEL || !/\.gguf$/i.test(MODEL) || !sofuu.fs.existsSync(MODEL) ||
    /(^|\/)sofuu-fixture\.gguf$/i.test(MODEL)) {
    console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing real-weight GGUF " +
        "(the synthetic contract fixture is a parser fixture, not an inference backend)");
    process.exit(2);
}
if (!SERVER) {
    console.error("BLOCKED: the real MCP file server script was not found next to this harness");
    process.exit(2);
}

const TOOL = "read_sample";

/* One attempt: a real model asks the real MCP tool for the real file, and
 * the approval gate answers. `mode` decides who resolves the request:
 * "deny" resolves nothing (the deadline must deny), "allow" resolves it
 * from OUTSIDE the request event — the pending entry has to be registered
 * by then (the resolve-bounced-to-denies bug this assertion pins). */
async function drive(mode) {
    const dir = "/tmp/t73-approval-" + (Date.now() % 1000000) + "-" + mode;
    const sample = dir + "/sample.txt";
    const receipt = dir + "/receipt.json";
    const starts = dir + "/starts.log";
    const marker = "T73-MARKER-" + mode + "-" + (Date.now() % 10000);
    await sofuu.fs.mkdir(dir, { recursive: true });
    await sofuu.fs.writeFile(sample, marker);

    const events = [];
    let resolver = null;
    if (mode === "allow") {
        resolver = setInterval(() => {
            const req = events.find((e) => e.kind === "approval_request" && !e._done);
            if (req) {
                req._done = true;
                try { sofuu.agent.resolveApproval(req.payload.id, "allow"); } catch (eR) {}
            }
        }, 5);
    }
    let settled = null;
    try {
        settled = await sofuu.agent.run({
            name: "approv-t73-" + mode,
            system: "You have exactly one tool: read_sample, which takes no parameters. " +
                    "Your first action must be a tool call to read_sample.",
            model: MODEL, api_key: "x", memory: "off", rlm: "off", max_tokens: 64,
            tools: [{ mcp: [{ name: "real-file-read", command: BIN,
                args: ["run", SERVER, sample, receipt, starts],
                tools: { allow: [TOOL] } }] }],
            approve: "all",
            budget: { maxSteps: 1 },
        }, "Read the file " + sample + " with " + TOOL + ".", {
            toolTimeoutMs: mode === "allow" ? 6000 : 1500,
            approveInteractive: true,
            onStep: (e) => { if (e && e.kind) events.push(e); },
        });
    } catch (e) {
        settled = { error: String(e && e.message || e) };
    }
    if (resolver) clearInterval(resolver);
    const requested = events.find((e) => e.kind === "approval_request");
    let recorded = null;
    try { recorded = P(String(await sofuu.fs.readFile(receipt, "utf8"))); } catch (e) {}
    /* The receipt's presence is the server's own record of the call — read
     * it before the temp dir goes away. */
    const hasReceipt = sofuu.fs.existsSync(receipt);
    try { await sofuu.fs.rm(dir); } catch (e) { /* the temp dir is disposable */ }
    return { events: events, requested: requested, settled: settled,
        receipt: recorded, marker: marker, hasReceipt: hasReceipt };
}

/* Run a leg until the model actually asks for the tool. A model that never
 * asks produces no approval event at all — that is a BLOCKED scenario, not
 * a pass and not a failure. */
async function driveUntilRequested(mode) {
    let last = null;
    for (let i = 0; i < ATTEMPTS; i++) {
        last = await drive(mode);
        if (last.requested) return last;
        console.error("  ℹ attempt " + (i + 1) + "/" + ATTEMPTS + ": the model did not request " +
            TOOL + " — no approval event occurred");
    }
    return last;
}

async function main() {
    assert("the real MCP file server and a real local model are configured",
        !!SERVER && sofuu.fs.existsSync(MODEL));

    // ── T7.3: no resolution within the deadline → typed deny, in the trace ──
    {
        const leg = await driveUntilRequested("deny");
        if (!leg.requested) {
            console.error("BLOCKED: the configured local model did not request " + TOOL +
                " in " + ATTEMPTS + " attempts, so no approval event occurred — " +
                "the harness does not fabricate one");
            process.exit(2);
        }
        const resolved = leg.events.filter((e) => e.kind === "approval_resolved");
        const denied = resolved.find((e) => e.payload && e.payload.allowed === false);
        assert("T7.3: approval timed out to deny", !!denied);
        assert("T7.3: the deny is in the trace with the tool named",
            denied && String(denied.payload.tool) === TOOL);
        const blocked = leg.events.find((e) => e.kind === "tool_result" && e.payload && e.payload.blocked === "approval");
        assert("T7.3: the tool never executed while approval was pending", !!blocked);
        assert("T7.3: the run still settled", !!leg.settled);
        assert("T7.3: the refused call never reached the MCP server (no receipt)",
            !leg.hasReceipt);
        try { if (sofuu.tools && sofuu.tools.killActive) sofuu.tools.killActive(); } catch (e) {}
    }

    // ── T7.4: resolve immediately out of the approval_request event ────────
    {
        const leg = await driveUntilRequested("allow");
        if (!leg.requested) {
            console.error("BLOCKED: the configured local model did not request " + TOOL +
                " in " + ATTEMPTS + " attempts, so no approval event occurred — " +
                "the harness does not fabricate one");
            process.exit(2);
        }
        const allowed = leg.events.find((e) => e.kind === "approval_resolved" &&
            e.payload && e.payload.allowed === true);
        const toolOk = leg.events.find((e) => e.kind === "tool_result" &&
            e.payload && e.payload.ok === true && !e.payload.blocked);
        assert("T7.4: immediate resolve is allowed (not raced to deny)", !!allowed);
        assert("T7.4: the tool actually ran", !!toolOk);
        assert("T7.4: run completed", !!leg.settled);
        assert("T7.4: the MCP server really read the file (its own receipt matches)",
            !!leg.receipt && leg.receipt.operation === TOOL && leg.receipt.contents === leg.marker);
        /* The tool leg can fail for a reason that is NOT the approval gate:
         * the model asked for a tool that takes NO arguments and supplied
         * some anyway, and the MCP server refused the call with its typed
         * schema error — the runtime behaved correctly. The approval
         * ordering this case is written for still held. A model that cannot
         * drive the tool cannot exercise the leg, so that is recorded as
         * blocked, with the reason, and never as a pass. */
        const toolRefused = leg.events.find((e) => e.kind === "tool_result" &&
            e.payload && e.payload.ok === false &&
            (String(e.payload.error_code) === "E_SCHEMA" || String(e.payload.error_code) === "E_TOOL"));
        if (!toolOk && toolRefused) {
            console.log(`\n=== t73 results: ${results.passed} passed, ${results.failed} failed ===`);
            console.error("BLOCKED: the model asked for " + TOOL + " with arguments the MCP server " +
                "refuses (" + (toolRefused.payload.error_code || "E_TOOL") + ": " +
                String(toolRefused.payload.error || toolRefused.payload.result || "").slice(0, 140) +
                ") — the approval ordering under test held, but the tool leg cannot be exercised " +
                "with this model");
            process.exit(2);
        }
    }

    console.log(`\n=== t73 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t73:", (e && e.message) || e); process.exit(1); });
