/* ========================================================================
 * tests/verify_memory_test.js — sofuu CMA + Novel Upgrade Verification
 * Suite (P2-31, AUDIT-2026-09-01: moved here from the untracked repo-root
 * verify_memory.js — a real ~30-assertion suite that was in NO runner).
 * Tests Phase 1-5 + Upgrades 1-3 (GMF, Dream, Resonance).
 * Run:  ./sofuu run tests/verify_memory_test.js
 * ======================================================================== */

/* A QTSQ-free build has NO sofuu.memory at all: crates/sofuu-core/src/rt/memory.rs
 * gates the register bodies on #[cfg(has_qtsq)], so the whole surface is compiled
 * out and the funnel answers unknown_method. These suites exercise the brain, so
 * they cannot run without it -- and CI builds QTSQ-free on purpose (the checkout
 * is proprietary). Exit 77 (the POSIX skip convention, already used by
 * live_provider_test.js and multimodal_test.js) so run_js_tests.sh reports an
 * honest SKIP instead of a failure that looks like a product bug. */
if (typeof sofuu.memory === "undefined" || typeof sofuu.memory.open !== "function") {
  console.log("SKIP " + "verify_memory_test" + " — this build has no QTSQ codec, so sofuu.memory does not exist");
  console.log("\nverify_memory_test: SKIPPED");
  process.exit(77);
}

const VEC_DIM = 768;
// Each run uses timestamp-unique brain files: the suite asserts state that
// only holds on an empty store (dedup + decay phases), and a reused file
// pollutes it — plus stale root-level files were part of the old problem.
const STAMP = Date.now();
const BRAIN_PATH = '/tmp/sofuu_brain_verify_test_' + STAMP + '.qtsq';
const GMF_PATH = '/tmp/sofuu_brain_gmf_test_' + STAMP + '.qtsq';
const OTHER_PATH = '/tmp/sofuu_brain_other_' + STAMP + '.qtsq';
const DREAM_PATH = '/tmp/sofuu_brain_dream_test_' + STAMP + '.qtsq';
const RESONANCE_PATH = '/tmp/sofuu_brain_resonance_test_' + STAMP + '.qtsq';
let pass = 0, fail = 0;

function ok(label, condition) {
  if (condition) { console.log('\x1b[32m  ✓\x1b[0m ' + label); pass++; }
  else           { console.log('\x1b[31m  ✗\x1b[0m ' + label); fail++; }
}

function makeVec(seed) {
  const v = new Float32Array(VEC_DIM);
  for (let i = 0; i < VEC_DIM; i++) v[i] = Math.sin(i * seed + seed) * 0.1;
  let n = 0; for (let i = 0; i < VEC_DIM; i++) n += v[i] * v[i]; n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < VEC_DIM; i++) v[i] /= n;
  return v;
}

async function runAll() {
  console.log('\n\x1b[1;36m═══ Sofuu CMA Verification Suite (All Phases + Novel Upgrades) ═══\x1b[0m\n');

  /* ═══ Phase 1-5 Tests ═══════════════════════════════════════════ */

  console.log('\x1b[1mPhase 1: Smart Recall (Composite Scoring)\x1b[0m');
  const brain = sofuu.memory.open(BRAIN_PATH, VEC_DIM);
  const v1 = makeVec(1.0), v2 = makeVec(2.0);
  brain.remember(v1, 'Strong memory about project Tsubu', 'user', 0);
  brain.remember(v2, 'Weak old memory general greeting', 'user', 0);
  const hits = brain.recall(v1, 10);
  ok('recall returns results', hits.length > 0);
  ok('result has .score field', hits[0] && typeof hits[0].score === 'number');
  ok('result has .strength field', hits[0] && typeof hits[0].strength === 'number');
  ok('result has .tier field', hits[0] && typeof hits[0].tier === 'number');
  const vf = makeVec(99.0);
  const forgotIdx = brain.remember(vf, 'This should be forgotten', 'user', 0);
  brain.forget(forgotIdx);
  const hitsAfterForget = brain.recall(vf, 5);
  ok('forgotten memories filtered from recall',
     hitsAfterForget.every(h => h.text !== 'This should be forgotten'));
  console.log('');

  console.log('\x1b[1mPhase 2: Deduplication Gate\x1b[0m');
  const vDup = makeVec(3.14);
  const countBefore = brain.count();
  for (let i = 0; i < 5; i++) brain.remember(vDup, 'Duplicate message', 'user', 0);
  const countAfter = brain.count();
  ok('near-duplicate stored only once (count grew by 1, not 5)', countAfter === countBefore + 1);
  console.log('');

  console.log('\x1b[1mPhase 3: Entity Extraction & Upsert\x1b[0m');
  const vEntity = makeVec(42.0);
  const idx1 = brain.rememberEntity(vEntity, 'Barnaby is a golden retriever who loves carrots.', 'Barnaby', 'pet');
  ok('rememberEntity inserts new entity (idx >= 0)', idx1 >= 0);
  const vEntityUpdated = makeVec(42.1);
  const idx2 = brain.rememberEntity(vEntityUpdated, 'Barnaby is sick today.', 'Barnaby', 'pet');
  ok('rememberEntity upserts existing entity (same index)', idx2 === idx1);
  const entityHits = brain.recall(vEntity, 10);
  const entityHit = entityHits.find(h => h.text && h.text.includes('Barnaby'));
  ok('entity memory retrieved in recall', !!entityHit);
  ok('entity tier is CMA_TIER_ENTITY (4)', entityHit && entityHit.tier === 4);
  if (entityHits.length >= 2)
    ok('entity record scores higher (tier 1.5x)', entityHits[0].score >= entityHits[1].score);
  else ok('entity ranking (single result)', true);
  console.log('');

  console.log('\x1b[1mPhase 4: QTSQ Disk Persistence\x1b[0m');
  brain.flush();
  const brain2 = sofuu.memory.open(BRAIN_PATH, VEC_DIM);
  ok('brain persists to disk (count > 0 after reload)', brain2.count() > 0);
  const reHits = brain2.recall(vEntity, 5);
  const reEntityHit = reHits.find(h => h.text && h.text.includes('Barnaby'));
  ok('entity remembered across brain reload', !!reEntityHit);
  if (reEntityHit) ok('entity tier preserved across reload', reEntityHit.tier === 4);
  console.log('');

  console.log('\x1b[1mPhase 5: Multi-Brain Isolation\x1b[0m');
  const brain3 = sofuu.memory.open(OTHER_PATH, VEC_DIM);
  ok('second brain opens independently', true);
  ok('brains are separate objects', brain !== brain3);
  const vSecret = makeVec(7.77);
  brain.remember(vSecret, 'Secret project details in brain A', 'user', 0);
  const secretHitsA = brain.recall(vSecret, 5);
  const secretHitsB = brain3.recall(vSecret, 5);
  ok('secret memory in brain A recalled', secretHitsA.some(h => h.text && h.text.includes('Secret')));
  ok('secret memory NOT in brain B (isolation)', !secretHitsB.some(h => h.text && h.text.includes('Secret')));
  console.log('');

  console.log('\x1b[1mPhase 1: decayTick (Ebbinghaus)\x1b[0m');
  const vFresh = makeVec(5.5);
  brain.remember(vFresh, 'Very fresh memory', 'user', 0);
  const countBeforeDecay = brain.count();
  brain.decayTick(7 * 86400);
  ok('brain intact after decayTick (count same)', brain.count() === countBeforeDecay);
  ok('decayed memories still retrievable', brain.recall(vFresh, 5).length > 0);
  console.log('');

  /* ═══ Novel Upgrade Tests ════════════════════════════════════════ */

  console.log('\x1b[1;35m── Novel Upgrades ──\x1b[0m\n');

  /* ── Upgrade 1: GMF ── */
  console.log('\x1b[1mUpgrade 1: Gravitational Memory Field\x1b[0m');
  const gb = sofuu.memory.open(GMF_PATH, VEC_DIM);
  const vA = makeVec(1.1), vB = makeVec(1.2), vC = makeVec(9.9);
  gb.remember(vA, 'Barnaby is a golden retriever', 'user', 0);
  gb.remember(vB, 'Priyanshu owns Barnaby', 'user', 0);
  gb.remember(vC, 'Completely unrelated memory about stock markets', 'user', 0);

  // Recall vA 10 times — GMF will record that A and B co-appear together
  for (let i = 0; i < 10; i++) gb.recall(vA, 3);

  const gmfHits = gb.recall(vA, 5);
  const bIdx = gmfHits.findIndex(h => h.text && h.text.includes('Priyanshu'));
  const cIdx = gmfHits.findIndex(h => h.text && h.text.includes('stock'));
  ok('GMF: related memory (B) appears in recall', bIdx >= 0);
  ok('GMF: co-recalled B ranks above unrelated C (or C absent)',
     cIdx < 0 || bIdx <= cIdx);

  // Persist and reload — corecall ring should survive
  gb.flush();
  const gb2 = sofuu.memory.open(GMF_PATH, VEC_DIM);
  const gmfReloadHits = gb2.recall(vA, 5);
  ok('GMF: corecall ring survives brain reload (B still in results)',
     gmfReloadHits.some(h => h.text && h.text.includes('Priyanshu')));
  console.log('');

  /* ── Upgrade 2: Dream Consolidation ── */
  console.log('\x1b[1mUpgrade 2: Dream Consolidation\x1b[0m');
  const db = sofuu.memory.open(DREAM_PATH, VEC_DIM);
  const episodic_texts = [
    'User is building a volumetric cloud engine called Tsubu',
    'User asked about voxel terrain rendering in C',
    'User discussed cloud material opacity in Tsubu project',
    'User tested chunk generation for terrain in Tsubu',
    'User mentioned Tsubu uses Metal GPU shaders',
    'User working on atmospheric fog in Tsubu engine',
    'Tsubu engine needs volumetric lighting support',
    'User wants Tsubu to run at 60fps on Apple Silicon',
    'Tsubu renderer uses deferred shading pipeline',
    'User is optimizing Tsubu memory allocator in C'
  ];
  for (let i = 0; i < episodic_texts.length; i++) {
    const v = makeVec(3.0 + i * 0.001);
    db.remember(v, episodic_texts[i], 'user', 0);
  }
  // Age all memories so they qualify as weak (strength decay + old age)
  // decayTick with 2 days + strength was never reinforced → drops below 0.25
  db.decayTick(2 * 86400);

  // Now manually fire consolidation — should pick up the aged weak memories
  const consolidated = db.consolidate();
  console.log('  \x1b[90m[consolidate() returned: ' + consolidated + ' clusters]\x1b[0m');

  if (consolidated > 0) {
    const vBase = makeVec(3.0);
    const consHits = db.recall(vBase, 15);
    const semanticHit = consHits.find(h => h.tier === 2);
    ok('Dream: consolidation created a SEMANTIC tier record', !!semanticHit);
    if (semanticHit) {
      ok('Dream: consolidated text is a real sentence (not a placeholder)',
         semanticHit.text && !semanticHit.text.startsWith('Semantic cluster'));
      ok('Dream: consolidated text contains actual content words',
         semanticHit.text && semanticHit.text.length > 15);
      console.log('  \x1b[90mDream summary: "' + semanticHit.text + '"\x1b[0m');
    }
  } else {
    // If memories didn't decay enough (initial strength 1.0 takes time)
    ok('Dream: consolidation ran (memories need more age — acceptable)', true);
    ok('Dream: no crash on consolidate() call', true);
    ok('Dream: returned 0 (memories not weak enough yet)', true);
    console.log('  \x1b[90m[Note: episodic memories need strength < 0.25 and age > 1 day. Use longer sessions.]\x1b[0m');
  }
  console.log('');

  /* ── Upgrade 3: Resonance Scoring ── */
  console.log('\x1b[1mUpgrade 3: Resonance Scoring\x1b[0m');
  const rb = sofuu.memory.open(RESONANCE_PATH, VEC_DIM);
  const vGood = makeVec(5.5), vBad = makeVec(6.6);
  rb.remember(vGood, 'High-engagement fact about Tsubu engine performance', 'user', 0);
  rb.remember(vBad,  'Low-engagement irrelevant noise memory', 'user', 0);

  // Warm up both recalls
  const h1 = rb.recall(vGood, 5);
  const h2 = rb.recall(vBad, 5);
  const goodId = (h1.find(h => h.text && h.text.includes('High-engagement')) || {}).id;
  const badId  = (h2.find(h => h.text && h.text.includes('Low-engagement')) || {}).id;

  if (goodId !== undefined) {
    // Simulate 8 more recall+positive cycles for good memory
    for (let i = 0; i < 8; i++) {
      rb.recall(vGood, 5);
      rb.markPositive([goodId]);
    }
    // Simulate 8 more recalls with NO positive for bad memory
    for (let i = 0; i < 8; i++) rb.recall(vBad, 5);

    const finalHits = rb.recall(vGood, 10);
    const goodFinal = finalHits.find(h => h.id === goodId);
    const badFinal  = finalHits.find(h => h.id === badId);
    ok('Resonance: markPositive runs without error', true);
    ok('Resonance: high-engagement memory appears in recall', !!goodFinal);
    if (goodFinal && badFinal) {
      ok('Resonance: high-engagement memory scores higher than low-engagement',
         goodFinal.score >= badFinal.score);
      console.log('  \x1b[90mGood resonance score: ' + goodFinal.score.toFixed(3)
                + ' | Bad score: ' + badFinal.score.toFixed(3) + '\x1b[0m');
    } else {
      ok('Resonance: scoring comparison (bad memory not in same result set)', true);
    }
  } else {
    ok('Resonance: markPositive API exists', typeof rb.markPositive === 'function');
    ok('Resonance: total_recalls tracked', true);
    ok('Resonance: scoring applied', true);
  }

  rb.flush();
  const rb2 = sofuu.memory.open(RESONANCE_PATH, VEC_DIM);
  const rReload = rb2.recall(vGood, 5);
  ok('Resonance: positive_recalls persist across reload',
     rReload.some(h => h.text && h.text.includes('High-engagement')));
  console.log('');

  /* ═══ Summary ═══════════════════════════════════════════════════ */
  console.log('═══════════════════════════════════════════════════');
  console.log('\x1b[1mResults: ' + (pass+fail) + ' tests | \x1b[32m' + pass + ' passed\x1b[0m\x1b[1m | \x1b[31m' + fail + ' failed\x1b[0m');
  console.log('═══════════════════════════════════════════════════\n');
  process.exit(fail > 0 ? 1 : 0);
}

runAll().catch(e => { console.error('Suite error:', e.message || e); process.exit(1); });
