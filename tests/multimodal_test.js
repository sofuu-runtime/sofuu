/* ========================================================================
 * tests/multimodal_test.js — M1 multimodal (text↔image) end-to-end.
 *
 * Proves the joint-space story through public surfaces only:
 *   readFileBytes → ai.embedImage → memory.open(64, image space) →
 *   remember → recall with a SEM2 *text* vector over the image store.
 *
 * Skips (exit 77) when the IMG1 artifact isn't baked or QTSQ is absent,
 * mirroring the memory suite's auto-skip. No network, no keys.
 * Run:  ./sofuu run tests/multimodal_test.js
 * ======================================================================== */

const STORE = '/tmp/sofuu_mm_test_' + Date.now() + '.qtsq';
let pass = 0, fail = 0;

function ok(label, condition) {
  if (condition) { console.log('\x1b[32m  ✓\x1b[0m ' + label); pass++; }
  else           { console.log('\x1b[31m  ✗\x1b[0m ' + label); fail++; }
}

function norm(v) {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

function cos(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function runAll() {
  // Fixtures live next to this file (8×8 solid PNGs, 74 bytes each).
  const redBytes = await sofuu.fs.readFileBytes('tests/fixtures/mm_red.png');
  const blueBytes = await sofuu.fs.readFileBytes('tests/fixtures/mm_blue.png');
  ok('fixtures load as Uint8Array', redBytes instanceof Uint8Array && redBytes.length > 0);

  // 1–3. embedImage vectors: shape, unit norm, distinct colors differ.
  const rv = sofuu.ai.embedImage(redBytes);
  const bv = sofuu.ai.embedImage(blueBytes);
  ok('red embeds to Float32Array(64)', rv instanceof Float32Array && rv.length === 64);
  ok('red vector is unit', Math.abs(norm(rv) - 1.0) < 1e-4);
  ok('blue embeds to Float32Array(64)', bv instanceof Float32Array && bv.length === 64);
  ok('red vs blue differ (cos < 0.99)', cos(rv, bv) < 0.99);

  // 4. Garbage refuses with a TypeError (never silent garbage).
  let threw = false;
  try {
    sofuu.ai.embedImage(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
  } catch (e) { threw = /undecodable|unavailable/.test(String((e && e.message) || e)); }
  ok('garbage bytes throw', threw);

  // 5–8. Image store: remember captioned vectors, recall by TEXT query.
  // Throws when IMG1/QTSQ is unavailable → suite skip (exit 77).
  let store;
  try {
    store = sofuu.memory.open(STORE, 64, 'image-projector-v1');
  } catch (e) {
    console.log('SKIP: image store unavailable (' + ((e && e.message) || e) + ')');
    process.exit(77);
  }
  ok('image store opens (64-dim, img1 space)', !!store);
  store.remember(rv, 'solid red background', 'user', 0);
  store.remember(bv, 'solid blue background', 'user', 0);

  const qRed = sofuu.ai.embedLocalSemanticV2('red background');
  const hitsRed = store.recall(qRed, 3) || [];
  ok('text query "red background" hits red first',
    hitsRed.length > 0 && /red/.test(hitsRed[0].text || ''));

  const qBlue = sofuu.ai.embedLocalSemanticV2('blue background');
  const hitsBlue = store.recall(qBlue, 3) || [];
  ok('text query "blue background" hits blue first',
    hitsBlue.length > 0 && /blue/.test(hitsBlue[0].text || ''));

  console.log('');
  console.log('═══════════════════════════════════════════════════');
  console.log('\x1b[1mResults: ' + (pass+fail) + ' tests | \x1b[32m' + pass + ' passed\x1b[0m\x1b[1m | \x1b[31m' + fail + ' failed\x1b[0m');
  console.log('═══════════════════════════════════════════════════\n');
  process.exit(fail > 0 ? 1 : 0);
}

runAll().catch(e => { console.error('Suite error:', e.message || e); process.exit(1); });
