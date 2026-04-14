/**
 * think_demo.js — Demonstrates Sofuu's Thinking Capabilities
 *
 * Shows:
 * - Point 2: Reasoning patterns recalled from CMA to guide future answers
 * - Point 3: Multi-step think → answer loop with native qwen3 thinking
 */

import { embed, thinkAndAnswer, learnReasoning, recallReasoning } from "../src/js/think.js";

const DIM  = 768;
const MIND = "think_demo.qtsq";
const mem  = Sofuu.memory.open(MIND, DIM);

console.log("═══════════════════════════════════════════════════");
console.log("  Sofuu Thinking Layer Demo");
console.log("═══════════════════════════════════════════════════\n");

// ────────────────────────────────────────────────────────────────
// Point 2: Seed CMA with a known reasoning pattern
// (Simulates the agent having solved this type of problem before)
// ────────────────────────────────────────────────────────────────
console.log("── Point 2: Seeding a reasoning pattern ──\n");

await learnReasoning(
    mem,
    "How should I explain time complexity to a beginner?",
    "1. Start with intuition (why loops are slower). 2. Use a real example (finding a name in a list). 3. Introduce Big-O as a label, not a formula. 4. Show O(n) vs O(n²) visually.",
    "Time complexity measures how an algorithm slows as input grows. Think of it like searching a phone book — flipping one page at a time is O(n), using alphabetical binary search is O(log n)."
);

console.log("  ✓ Stored reasoning pattern: 'explain complexity to beginner'");
console.log("  ✓ CMA now knows HOW to approach explanation problems\n");

// ────────────────────────────────────────────────────────────────
// Point 3: Ask a related question — pattern gets recalled + native thinking used 
// ────────────────────────────────────────────────────────────────
console.log("── Point 3: Multi-step thinking on a new problem ──\n");

const q1 = "What is HNSW and why is it fast?";
console.log(`Question: "${q1}"\n`);

const { answer: a1, reasoning: r1 } = await thinkAndAnswer(mem, q1, {
    useNativeThink: false,   // explicit two-pass: fast + no fetch timeout
    verbose: true
});

console.log(`\nFinal Answer:\n${a1}\n`);
console.log("─".repeat(60));

// ────────────────────────────────────────────────────────────────
// Second question: verify reasoning pattern gets recalled
// ────────────────────────────────────────────────────────────────
console.log("\n── Second question (should recall first pattern) ──\n");

const q2 = "What is memory decay in AI systems?";
console.log(`Question: "${q2}"\n`);

const patterns = await recallReasoning(mem, q2, 3);
if (patterns.length > 0) {
    console.log(`  [evidence] CMA recalled ${patterns.length} reasoning pattern(s):`);
    patterns.forEach(p => console.log(`    → ${p.substring(0, 70)}...`));
    console.log();
}

const { answer: a2 } = await thinkAndAnswer(mem, q2, {
    useNativeThink: false,
    verbose: false
});

console.log(`Final Answer:\n${a2}\n`);
console.log("═══════════════════════════════════════════════════");
console.log("  Demo complete. Reasoning patterns are now stored.");
console.log("  Re-run to see the agent recall its own reasoning.");
console.log("═══════════════════════════════════════════════════");
