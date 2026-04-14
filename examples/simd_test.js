// examples/simd_test.js — SIMD vector operations test

const { ai } = sofuu;

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else       { console.error("  ❌ FAIL: " + label); results.failed++; }
}
function near(a, b, eps = 0.0001) { return Math.abs(a - b) < eps; }

console.log("\n=== SIMD Vector Operations Test ===\n");

// ── Test 1: dot product ──────────────────────────────────────
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([4, 3, 2, 1]);
const dot = ai.dot(a, b);
assert("dot([1,2,3,4], [4,3,2,1]) === 20", near(dot, 20));

// ── Test 2: identical vectors → cosine similarity = 1.0 ─────
const x = new Float32Array(128).fill(1.0);
const y = new Float32Array(128).fill(1.0);
const sim = ai.similarity(x, y);
assert("similarity(same, same) ≈ 1.0", near(sim, 1.0));

// ── Test 3: orthogonal vectors → cosine similarity = 0.0 ────
const ox = new Float32Array([1, 0, 0, 0]);
const oy = new Float32Array([0, 1, 0, 0]);
assert("similarity(orthogonal) ≈ 0.0", near(ai.similarity(ox, oy), 0.0));

// ── Test 4: opposite vectors → cosine similarity = -1.0 ─────
const pos = new Float32Array([1, 0, 0, 0]);
const neg = new Float32Array([-1, 0, 0, 0]);
assert("similarity(opposite) ≈ -1.0", near(ai.similarity(pos, neg), -1.0));

// ── Test 5: L2 distance ──────────────────────────────────────
const p = new Float32Array([0, 0, 0]);
const q = new Float32Array([3, 4, 0]);
assert("l2([0,0,0], [3,4,0]) === 5.0", near(ai.l2(p, q), 5.0));

// ── Test 6: self-distance = 0 ───────────────────────────────
const s = new Float32Array([1, 2, 3, 4, 5]);
assert("l2(v, v) === 0", near(ai.l2(s, s), 0.0));

// ── Test 7: large vector (OpenAI embedding size) ─────────────
const dim = 1536;
const va = new Float32Array(dim);
const vb = new Float32Array(dim);
for (let i = 0; i < dim; i++) { va[i] = Math.sin(i); vb[i] = Math.cos(i); }
const large_sim = ai.similarity(va, vb);
assert("1536-dim similarity in range [-1,1]", large_sim >= -1 && large_sim <= 1);

// ── Test 8: plain JS Array fallback ─────────────────────────
const arr_a = [1, 0, 0];
const arr_b = [1, 0, 0];
assert("plain Array similarity === 1.0", near(ai.similarity(arr_a, arr_b), 1.0));

// ── Benchmark: 1536-dim cosine ───────────────────────────────
const RUNS = 1000;
const t0 = Date.now();
for (let i = 0; i < RUNS; i++) ai.similarity(va, vb);
const ms = Date.now() - t0;
console.log(`\n  Benchmark: ${RUNS}x cosine(1536-dim) = ${ms}ms total, ${(ms/RUNS).toFixed(3)}ms/call`);
assert(`throughput < 1ms/call`, ms < RUNS); /* should be well under 1ms each */

console.log(`\n=== RESULTS ===`);
console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
if (results.failed === 0) {
    console.log("\n✅ SIMD — all tests PASSED\n");
} else {
    console.error(`\n❌ ${results.failed} test(s) failed`);
    process.exit(1);
}
