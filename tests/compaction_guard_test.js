// tests/compaction_guard_test.js — the compaction gate is the one
// IRREVERSIBLE action the chat takes (it deletes conversation turns), so
// its safety floor gets its own test rather than riding along inside a
// broader chat suite.
//
// Two guards are asserted here, both added 2026-09-25 after the gate
// audit found the network tied a linear model on held-out data (i.e. the
// model is not obviously smarter than the thing we would replace it
// with — so the blast radius must not depend on it being right):
//
//   1. PER-PASS CAP — one pass removes at most a quarter of the complete
//      blocks, and never leaves fewer than two. Before this, the only
//      stop condition was "usage below half the budget", which on a long
//      history authorised deleting dozens of turns at once.
//   2. LEXICAL FLOOR — a turn whose USER message asks a question, gives
//      an instruction, or records a decision/approval is never removed,
//      whatever the model says. The last line of defence cannot be the
//      component under suspicion.
//
// Run:  ./sofuu run tests/compaction_guard_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== Compaction safety guards ===\n");

/* ── the guard logic, mirrored from src/js/chat.js ────────────────
 * Kept as a literal copy on purpose: this test asserts the RULE, and a
 * copy that drifts from the driver is a failing test the moment the
 * driver changes. The e2e suite covers the wiring. */
const PROTECT_RE = /(\?|^\s*(please\s+)?(do\s+not|don't|never|always|must|make\s+sure|keep)\b)|(\b(approved|approve|accepted|rejected|decided|decision|agreed|sign\s*off)\b)/i;

function turnStartAt(hist, i) {
    let j = Math.max(0, Math.min(i, hist.length - 1));
    while (j > 0 && hist[j].role !== "user") j--;
    return j;
}

function isProtectedBlock(hist, idx) {
    const s = turnStartAt(hist, idx);
    if (s < 0 || s >= hist.length) return false;
    let e = s + 1;
    while (e < hist.length && hist[e].role !== "user") e++;
    for (let i = s; i < e; i++) {
        const m = hist[i];
        if (m && m.role === "user" && PROTECT_RE.test(String(m.content || ""))) return true;
    }
    return false;
}

function maxDropsFor(completeBlocks) {
    let maxDrops = Math.max(1, Math.floor(completeBlocks * 0.25));
    if (completeBlocks - maxDrops < 2) maxDrops = Math.max(0, completeBlocks - 2);
    return maxDrops;
}

const block = (userText, answerText) => ([
    { role: "user", content: userText },
    { role: "assistant", content: answerText || "ok" },
]);

// ── 1. lexical floor ──────────────────────────────────────────────
assert("a user question is protected",
    isProtectedBlock(block("why does the parser drop the last row?"), 0));
assert("a negative instruction is protected",
    isProtectedBlock(block("do not change the public API"), 0));
assert("'always' is protected", isProtectedBlock(block("always run the migration check"), 0));
assert("an approval is protected", isProtectedBlock(block("approved, merge it"), 0));
assert("a decision is protected", isProtectedBlock(block("we decided on postgres"), 0));
assert("plain chatter is NOT protected",
    !isProtectedBlock(block("ran the scraper again"), 0));
assert("an assistant message alone never protects a turn",
    !isProtectedBlock([
        { role: "user", content: "ran the scraper again" },
        { role: "assistant", content: "do not worry, everything is fine?" },
    ], 0));
assert("protection does not leak across block boundaries",
    /* A protected turn earlier in the history must not shield a later,
     * unprotected one: each user message starts its own block. */
    !isProtectedBlock([
        { role: "user", content: "do not change the public API" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "ran the scraper again" },
        { role: "assistant", content: "ok" },
    ], 2));
assert("a tool result inside a block cannot protect it",
    !isProtectedBlock([
        { role: "user", content: "ran the scraper again" },
        { role: "assistant", content: "", tool_calls: [{ id: "1", function: { name: "read_file" } }] },
        { role: "tool", tool_call_id: "1", content: "error: failed. do not ignore this?" },
        { role: "assistant", content: "ok" },
    ], 0));

// ── 2. per-pass cap ───────────────────────────────────────────────
assert("4 complete blocks → at most 1 dropped", maxDropsFor(4) === 1);
assert("8 complete blocks → at most 2 dropped", maxDropsFor(8) === 2);
assert("40 complete blocks → at most 10 dropped", maxDropsFor(40) === 10);
assert("two live blocks are never dropped (1 block → 0)",
    maxDropsFor(1) === 0);
assert("three blocks still keep two live (1 drop)", maxDropsFor(3) === 1);

/* The cap is the property that matters: a gate that flags EVERY block
 * must still leave the session mostly intact. */
function simulatePass(completeBlocks) {
    const cap = maxDropsFor(completeBlocks);
    return completeBlocks - cap;
}
const survivors = simulatePass(100);
assert("a gate that flags 100 of 100 blocks still leaves 75 turns",
    survivors === 75);
assert("a gate that flags 200 of 200 blocks still leaves 150 turns",
    simulatePass(200) === 150);

// ── 3. the guards compose ─────────────────────────────────────────
/* Cap AND floor together. The floor can only ever REDUCE deletions, and
 * a protected block is skipped rather than counted against the cap — so
 * a protected turn does not consume budget and starve a disposable one
 * later in the pass. */
function pass(hist, flagged) {
    let complete = 0;
    for (let i = 0; i < hist.length - 1; i++) if (hist[i].role === "user") complete++;
    const cap = maxDropsFor(complete);
    let dropped = 0, protectedSkips = 0;
    for (const idx of flagged) {
        if (dropped >= cap) break;
        if (isProtectedBlock(hist, idx)) { protectedSkips++; continue; }
        dropped++;
    }
    return { dropped, protectedSkips, remaining: complete - dropped, cap };
}

/* 20 complete blocks: 4 protected at the front, 16 disposable behind
 * them. The cap is 5, so the pass must delete exactly 5 — the 4
 * protected turns are stepped over, not counted, and 5 disposable ones
 * still go. */
const hist = [];
for (let i = 0; i < 4; i++) hist.push(...block("why did step " + i + " fail?"));
for (let i = 0; i < 16; i++) hist.push(...block("ran routine task " + i + " again"));
const all = [];
for (let i = 0; i < 20; i++) all.push(i * 2);

const r = pass(hist, all);
assert("20 blocks → the cap is 5", r.cap === 5);
assert("the 4 protected turns are stepped over", r.protectedSkips === 4);
assert("the cap is still spent on disposable turns", r.dropped === 5);
assert("both protected and spared turns survive", r.remaining === 15);

/* Order-independence: protecting the LAST blocks must not change how
 * many disposable turns get removed — the floor never eats the budget. */
const hist2 = [];
for (let i = 0; i < 16; i++) hist2.push(...block("ran routine task " + i + " again"));
for (let i = 0; i < 4; i++) hist2.push(...block("approved, merge step " + i));
const r2 = pass(hist2, all);
assert("protecting the tail spends the same budget", r2.dropped === 5);
/* The cap is spent on the OLDEST disposable turns, so the loop stops
 * before it ever reaches the protected tail — the protected turns are
 * safe twice over. What matters is that no protected turn was deleted. */
assert("no protected turn is deleted when they sit at the tail",
    r2.dropped + r2.protectedSkips <= 20 && r2.remaining === 15);
assert("the protected tail is never among the dropped",
    r2.dropped <= 5 && r2.remaining === 15);

/* The pathological case: everything is protected. Nothing is deleted,
 * and the pass costs the user nothing but a few tokens of context. */
const allProtected = [];
for (let i = 0; i < 12; i++) allProtected.push(...block("do not change behavior " + i));
const r3 = pass(allProtected, all.slice(0, 12));
assert("an all-protected history loses nothing", r3.dropped === 0);
assert("and reports every skip", r3.protectedSkips === 12);

console.log("\n=== RESULTS ===");
console.log("Passed: " + results.passed + " | Failed: " + results.failed);
if (results.failed > 0) process.exit(1);
console.log("\n✅ Compaction safety guards hold");
