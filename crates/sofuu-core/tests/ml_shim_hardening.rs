// tests/ml_shim_hardening.rs — Phase 1.2 of PLAN-ML-IMPROVEMENT-NO-RETRAINING
// (§5.2 "Harden every native ML boundary").
//
// Drives every registered sofuu.ml.* API through the REAL QuickJS engine
// with malformed, oversized, adversarial, and Unicode-hostile inputs and
// asserts the contract holds end to end:
//   * every return value parses as JSON (or a documented kind string);
//   * every score lands in [0, 1];
//   * every use/skip/keep/compact list holds valid, unique, in-range ids;
//   * no call aborts the turn (the JS throws → eval_string returns != 0).
//
// Engine tests serialize on rt::TEST_LOOP_LOCK inside the lib's own test
// binary (the runtime owns the process-global libuv loop). This is a
// SEPARATE test binary = a separate process with its own loop, so no
// cross-binary serialization is possible or needed.

use sofuu_core::ml::mod_ml_register;

#[test]
fn ml_boundaries_survive_malformed_and_hostile_inputs() {
    let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");
    let ctx = rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext;
    assert!(!ctx.is_null());
    // SAFETY: ctx is the live engine context, valid for the runtime's
    // lifetime; registration is idempotent per boot.
    unsafe { mod_ml_register(ctx) };

    // One JS program, many sub-checks: any contract break throws and the
    // eval returns non-zero (the turn must never abort quietly broken).
    let prog = r#"
(function () {
  var J = function (s) { return JSON.parse(s); };
  var probe = function (name, fn) {
    try { fn(); } catch (e) { throw new Error(name + ' threw: ' + e); }
  };
  var big = 'x'.repeat(200000);           // oversized text
  var uni = '配列のテスト😀' + '\u0000'.repeat(10) + '\uD800'; // unicode + NULs + lone surrogate
  var deep = JSON.stringify({ a: { b: { c: { d: big } } } });

  // ── freshness.score: malformed args, giant text, hostile opts ──────
  probe('freshness/no-args', function () { sofuu.ml.freshness.score(); });
  probe('freshness/nonstring', function () { sofuu.ml.freshness.score(null, undefined, 42); });
  probe('freshness/malformed-opts', function () { sofuu.ml.freshness.score('text', 'task', '{not json'); });
  probe('freshness/inf-opts', function () {
    var v = J(sofuu.ml.freshness.score(big, uni, '{"strength":1e400,"ageDays":-5,"kind":"weird"}'));
    if (!(v.score >= 0 && v.score <= 1)) throw new Error('score out of [0,1]: ' + v.score);
    if (Number.isNaN(v.score) || v.score === undefined) throw new Error('NaN score');
  });
  probe('freshness/quote-reason', function () {
    // A year-bearing text with quotes/newlines in the evidence path must
    // still parse (reason is escaped).
    var v = J(sofuu.ml.freshness.score('quoted "evidence"\nline dated 2019', 'latest "version"?'));
    if (typeof v.reason !== 'string') throw new Error('reason must be a string');
  });

  // ── supervisor.check / loop: garbage JSON, wrong types, huge fields ─
  probe('supervisor/empty', function () { J(sofuu.ml.supervisor.check('{}')); });
  probe('supervisor/malformed', function () { sofuu.ml.supervisor.check('['); });
  probe('supervisor/nonobject', function () { J(sofuu.ml.supervisor.check('"just a string"')); });
  probe('supervisor/hostile', function () {
    var v = J(sofuu.ml.supervisor.check(JSON.stringify({
      run: big.slice(0, 100000), step: 4294967296, tool: uni, sig: big.slice(0, 100000),
      target: uni, argsText: big, skipTargets: [uni, big, '', 'ok', null, 42],
      budget: -1, task: deep
    })));
    if (!(v.score >= 0 && v.score <= 1)) throw new Error('score out of [0,1]');
  });
  probe('supervisor/loop-malformed', function () { sofuu.ml.supervisor.loop('nope'); });

  // ── relevance.plan: giant menus, mixed-shape rows, kept ids OOB ─────
  probe('relevance/empty', function () { J(sofuu.ml.relevance.plan('{}')); });
  probe('relevance/malformed', function () { sofuu.ml.relevance.plan('{'); });
  probe('relevance/hostile-menu', function () {
    var cands = [];
    for (var i = 0; i < 500; i++) cands.push({ text: (i % 7 === 0) ? uni : big.slice(0, 50000), kind: (i % 3 === 0) ? 7 : 'memory', strength: 1e400, role: -3, path: 42 });
    cands.push(null); cands.push('a string'); cands.push({ text: undefined });
    var v = J(sofuu.ml.relevance.plan(JSON.stringify({
      task: big.slice(0, 9000), recent: uni, candidates: cands
    }), '{"kept":[0,3,999999,-1,null,"x"]}'));
    var seen = {};
    v.use.concat(v.skip).forEach(function (id) {
      if (!Number.isInteger(id) || id < 0 || id >= 64) throw new Error('id out of range: ' + id);
      if (seen[id]) throw new Error('duplicate id: ' + id);
      seen[id] = true;
    });
    v.scores.forEach(function (s) {
      if (!(s >= 0 && s <= 1)) throw new Error('score out of [0,1]: ' + s);
    });
  });

  // ── compaction.plan: giant histories, bad kinds, huge budgets ──────
  probe('compaction/empty', function () { J(sofuu.ml.compaction.plan('{}')); });
  probe('compaction/malformed', function () { sofuu.ml.compaction.plan(')'); });
  probe('compaction/hostile-segs', function () {
    var segs = [];
    for (var i = 0; i < 600; i++) segs.push({ text: (i % 5 === 0) ? uni : big.slice(0, 30000), tokens: 4294967296 + i, age: -5, kind: 9, retrievable: 'yes', compacted: 3 });
    var v = J(sofuu.ml.compaction.plan(JSON.stringify({
      task: big.slice(0, 9000), summary: uni, recent: big.slice(0, 30000), segments: segs
    }), '{"budget": 1e19}'));
    var seen = {};
    v.compact.concat(v.keep).forEach(function (id) {
      if (!Number.isInteger(id) || id < 0 || id >= 256) throw new Error('id out of range: ' + id);
      if (seen[id]) throw new Error('duplicate id: ' + id);
      seen[id] = true;
    });
    v.scores.forEach(function (s) { if (!(s >= 0 && s <= 1)) throw new Error('score out of [0,1]: ' + s); });
  });

  // ── alloc.plan / noteLimit / ingestListing ─────────────────────────
  probe('alloc/empty', function () { J(sofuu.ml.alloc.plan('{}')); });
  probe('alloc/malformed', function () { sofuu.ml.alloc.plan('}{'); });
  probe('alloc/hostile-state', function () {
    var v = J(sofuu.ml.alloc.plan(JSON.stringify({
      model: uni, baseUrl: big.slice(0, 40000), cfgWindow: -100, cfgMaxOutput: 1e15,
      overheadTk: -5000, historyTk: 1e18, turns: 1e12, toolFrac: 42,
      growthTk: 1e400, growthAccel: -7, taskTk: -3, attachTk: 1e18,
      meanAnswerTk: NaN, maxAnswerTk: 1e400
    })));
    if (!(v.window > 0 && v.maxOutput > 0)) throw new Error('plan must stay positive');
    if (!(v.pressure >= 0 && v.pressure <= 1)) throw new Error('pressure out of [0,1]');
    if (!(v.compactAt >= 0.5 && v.compactAt <= 0.7)) throw new Error('compactAt out of band');
    if (typeof v.source !== 'string') throw new Error('source must be a string');
    v.notes.forEach(function (n) { if (typeof n !== 'string') throw new Error('note not a string'); });
  });
  probe('alloc/noteLimit-garbage', function () {
    if (sofuu.ml.alloc.noteLimit('', '') !== '') throw new Error('empty model learns nothing');
    if (sofuu.ml.alloc.noteLimit(big.slice(0, 10000), big) !== '') {
      // A capped 10000-char model name cannot match a stored key — nothing learned.
      throw new Error('runaway model id must not learn');
    }
  });
  probe('alloc/ingest-malformed', function () {
    // Returns the count as a STRING ("-1" on malformed JSON — the caller
    // surfaces its own fetch error), per the shim's JSON-string contract.
    if (sofuu.ml.alloc.ingestListing('https://x.example/v1', '{bad json') !== '-1') throw new Error('malformed listing must return "-1"');
  });
})();
"#;
    let rc = rt.eval_string(prog, "<ml-shim-hardening>");
    assert_eq!(rc, 0, "a native ML boundary broke under hostile input — see the thrown Error text above");
}
