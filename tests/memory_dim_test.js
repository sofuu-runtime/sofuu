/* ========================================================================
 * tests/memory_dim_test.js — `sofuu.memory` accepts ARBITRARY embedding
 * dimensions, and refuses anything above its documented cap CLEANLY (a
 * thrown, coded error — never a crash).
 *
 * The brain used to be hard-wired enough around the 768d chat geometry that
 * an unusual embedding width was a coin flip: some widths silently stored
 * garbage or could not be reopened. This gate pins the contract on the two
 * ends of the range:
 *
 *   • dims 1, 2, 7, 16, 384, 1024 — a FULL round trip each:
 *         open → remember → count → flush → close → reopen → count → recall
 *     and the recalled top hit must be the exact text that was stored.
 *     (384 is the semantic-model width, 1024 the hash-embedder width, the
 *     rest are widths no baked model covers: the legacy-dim-N space.)
 *   • dim 8192 — the documented cap, INCLUSIVE: the same round trip must
 *     work at the boundary, otherwise "the cap is 8192" is a lie in the
 *     direction that matters least.
 *   • dim 8193 — one past the cap: `open` must THROW (code E_CORRUPT,
 *     message "could not be opened") and the process must keep running
 *     (the later assertions in this file are the liveness proof).
 *
 * Run:  ./sofuu run tests/memory_dim_test.js
 * (A QTSQ-free build has no sofuu.memory at all — rt/memory.rs gates the
 * register bodies on #[cfg(has_qtsq)] — so it exits 77 = the POSIX skip
 * convention run_js_tests.sh reports as SKIP, never as a pass. Not yet in
 * that registry; run this file directly until it is added there.)
 * ======================================================================== */

if (typeof sofuu.memory === "undefined" || typeof sofuu.memory.open !== "function") {
  console.log("SKIP memory_dim_test — this build has no QTSQ codec, so sofuu.memory does not exist");
  process.exit(77);
}

const M = sofuu.memory;
/* A unique temp DIRECTORY per run (one brain file per dim lives inside it),
 * so a rerun never inherits a previous run's store or lock file. */
const STAMP = Date.now() + '_' + Math.floor(Math.random() * 1e9);
const DIR = '/tmp/sofuu_dim_test_' + STAMP;

const ROUND_TRIP_DIMS = [1, 2, 7, 16, 384, 1024];
const CAP_DIM = 8192;          // rt/memory.rs: `vec_dim > 8192` is refused
const OVER_CAP_DIM = CAP_DIM + 1;

let pass = 0, fail = 0;

function ok(label, cond) {
  if (cond) { console.log('\x1b[32m  ✓\x1b[0m ' + label); pass++; }
  else      { console.log('\x1b[31m  ✗\x1b[0m ' + label); fail++; }
}

/* Run fn, return the thrown error (or null). */
function err(fn) {
  try { fn(); return null; } catch (e) { return e; }
}

function msgOf(e) { return e ? String(e.message || e) : ''; }

/* Deterministic, non-degenerate vector of ANY width. Values live in
 * [0.125, 0.875] so even dim 1 never sends the zero vector (a cosine
 * against a zero query is a degenerate answer, not a dimension bug). */
function vecFor(dim, seed) {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = (((i + seed) % 7) + 1) / 8;
  return v;
}

/* sofuu.fs.rm is NOT recursive — walk the tree, files first. */
async function rmrf(p) {
  const names = await sofuu.fs.readdir(p).catch(function () { return null; });
  if (names) {
    for (let i = 0; i < names.length; i++) await rmrf(p + '/' + names[i]);
  }
  await sofuu.fs.rm(p).catch(function () {});
}

/* open → remember → count → flush → close → reopen → count → recall.
 * Every step is asserted; `expectCount` is what the count must read on both
 * sides of the reopen (the store must not lose, duplicate or invent rows). */
function roundTrip(dim, opts) {
  opts = opts || {};
  const path = DIR + '/brain_' + (opts.tag || dim) + '.qtsq';
  const text = 'dim ' + dim + ' fact ' + STAMP;
  const v = vecFor(dim, dim + 1);
  const n = opts.records || 1;

  const b = M.open(path, dim);
  ok(dim + 'd: open returns a live handle', !!b && typeof b.remember === 'function');
  ok(dim + 'd: the store starts empty', b.count() === 0);

  const ids = [];
  for (let i = 0; i < n; i++) {
    ids.push(b.remember(i === 0 ? v : vecFor(dim, dim + 1 + i), i === 0 ? text : text + ' #' + i, 'user', 0));
  }
  ok(dim + 'd: remember() accepts a ' + dim + '-wide Float32Array (id ' + ids[0] + ')',
     ids.every(function (id) { return Number.isInteger(id) && id >= 0; }));
  ok(dim + 'd: count() right after remember is ' + n, b.count() === n);

  const f = b.flush();
  ok(dim + 'd: flush() sealed the records (' + (f && f.recordsWritten) + ' written)',
     f && f.ok === true && f.recordsWritten === n);
  ok(dim + 'd: close() releases the handle', M.close(b) === true && M.handles() === 0);

  const b2 = M.open(path, dim);
  ok(dim + 'd: the brain REOPENS at its own width', !!b2);
  ok(dim + 'd: count() survives flush+reopen (' + (b2 && b2.count()) + '/' + n + ')',
     b2 && b2.count() === n);

  const hits = b2.recall(v, 3);
  ok(dim + 'd: recall() answers with a hit', Array.isArray(hits) && hits.length >= 1);
  ok(dim + 'd: the top hit is the record that was written',
     Array.isArray(hits) && hits.length >= 1 && hits[0] && hits[0].text === text);
  ok(dim + 'd: the top hit is an exact match (distance ~0)',
     Array.isArray(hits) && hits.length >= 1 && hits[0] && typeof hits[0].distance === 'number' &&
     hits[0].distance < 1e-4);
  ok(dim + 'd: the hit keeps its role', Array.isArray(hits) && hits.length >= 1 &&
     hits[0] && hits[0].role === 'user');

  M.close(b2);
  ok(dim + 'd: close() after reopen releases the handle', M.handles() === 0);
}

async function main() {
  console.log('\n\x1b[1;36m═══ sofuu.memory arbitrary embedding dims (1 … 8192, cap 8192) ═══\x1b[0m\n');

  await sofuu.fs.mkdir(DIR, { recursive: true });
  ok('the per-run temp dir exists (' + DIR + ')', await sofuu.fs.exists(DIR));

  try {
    /* ═══ unusual widths: a full round trip each ═════════════════════ */
    console.log('\x1b[1mArbitrary dims — open/remember/flush/close/reopen/count/recall\x1b[0m');
    for (let i = 0; i < ROUND_TRIP_DIMS.length; i++) roundTrip(ROUND_TRIP_DIMS[i]);

    /* ═══ the documented cap is INCLUSIVE ════════════════════════════ */
    console.log('\n\x1b[1mCap boundary — ' + CAP_DIM + ' must still work\x1b[0m');
    roundTrip(CAP_DIM, { tag: 'cap', records: 2 });

    /* ═══ one past the cap: refused cleanly, process survives ════════ */
    console.log('\n\x1b[1mAbove the cap — ' + OVER_CAP_DIM + ' is REFUSED (no crash)\x1b[0m');
    const overPath = DIR + '/brain_over_cap.qtsq';
    const eOver = err(function () { M.open(overPath, OVER_CAP_DIM); });
    ok(OVER_CAP_DIM + 'd: open THROWS instead of returning a handle', !!eOver);
    ok(OVER_CAP_DIM + 'd: the refusal carries a code (E_CORRUPT)', eOver && eOver.code === 'E_CORRUPT');
    ok(OVER_CAP_DIM + 'd: the message explains the open failed',
       /could not be opened/.test(msgOf(eOver)));
    ok(OVER_CAP_DIM + 'd: no handle was leaked by the refusal', M.handles() === 0);
    ok(OVER_CAP_DIM + 'd: the refusal wrote no brain to disk', !(await sofuu.fs.exists(overPath)));

    /* The runtime is still healthy after the refusal: an open at the cap
     * works, and this process reaches the results block (a panic/abort
     * would have ended it here). */
    const afterPath = DIR + '/brain_after_refusal.qtsq';
    const after = M.open(afterPath, 16);
    const afterId = after.remember(vecFor(16, 99), 'after the refusal', 'user', 0);
    ok('a 16d brain still opens after the refusal (id ' + afterId + ')',
       Number.isInteger(afterId) && afterId >= 0);
    ok('the refused store and the live store stay independent', after.count() === 1);
    M.close(after);

    const eZero = err(function () { M.open(DIR + '/brain_zero.qtsq', 0); });
    ok('0 is refused too (not a silent degenerate store)', !!eZero && !!eZero.code);
  } finally {
    await rmrf(DIR);
  }

  ok('the temp dir was cleaned up', !(await sofuu.fs.exists(DIR)));

  console.log('');
  console.log('═══════════════════════════════════════════════════');
  console.log('\x1b[1mResults: ' + (pass + fail) + ' tests | \x1b[32m' + pass +
              ' passed\x1b[0m\x1b[1m | \x1b[31m' + fail + ' failed\x1b[0m');
  console.log('═══════════════════════════════════════════════════\n');
  process.exit(fail > 0 ? 1 : 0);
}

main().catch(function (e) {
  console.error('FAIL exception: ' + (e && e.stack || e));
  process.exit(1);
});
