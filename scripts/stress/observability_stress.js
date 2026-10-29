// scripts/stress/observability_stress.js — adversarial stress suite for the
// observability substrate (OBSERVABILITY_PLAN.md, docs/OBSERVABILITY-SPEC.md).
//
// This is not a unit test: every case drives the SHIPPED sofuu.events /
// sofuu.observability surface through the binary and asserts on what actually
// reached the disk. It needs no model, no network and no downloads.
//
//   ./sofuu run scripts/stress/observability_stress.js
//   make observability-stress
//
// The substrate is landing in parallel with this lane, so a missing method is
// a clean FAIL for its case, never a crash: each case checks its prerequisites
// and reports what is absent. Exits non-zero on any FAIL.
(function () {
  'use strict';

  var E = (typeof sofuu !== 'undefined' && sofuu.events) || null;
  var O = (typeof sofuu !== 'undefined' && sofuu.observability) || null;
  var FS = (typeof sofuu !== 'undefined' && sofuu.fs) || null;

  if (!FS || typeof FS.readFile !== 'function') {
    // The suite's whole point is proving what is ON DISK, so without the raw
    // read there is nothing to assert against.
    console.error('FATAL: sofuu.fs.readFile is not loaded in this build');
    if (typeof process !== 'undefined' && process.exit) process.exit(2);
    return;
  }

  var now = (typeof performance !== 'undefined' && performance && typeof performance.now === 'function')
    ? function () { return performance.now(); }
    : function () { return Date.now(); };

  var R = { cases: [], passed: 0, failed: 0, skipped: 0, startedAt: Date.now() };

  function record(name, ok, detail, ms) {
    R.cases.push({ name: name, ok: ok ? 1 : 0, detail: String(detail == null ? '' : detail), ms: Math.round(ms || 0) });
    if (ok) R.passed++; else R.failed++;
    console.log('  ' + (ok ? 'PASS' : 'FAIL') + ' ' + name + (detail ? '  [' + detail + ']' : '') + '  ' + Math.round(ms || 0) + 'ms');
  }
  function skip(name, why) { R.skipped++; console.log('  SKIP ' + name + '  [' + why + ']'); }

  async function check(name, fn) {
    var t0 = now();
    try { var d = await fn(); record(name, true, d, now() - t0); }
    catch (e) { record(name, false, (e && e.message) || e, now() - t0); }
  }
  function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }
  function eq(a, b, msg) {
    if (a !== b) throw new Error((msg || 'mismatch') + ': expected ' + JSON.stringify(b) + ' got ' + JSON.stringify(a));
  }
  function brief(v, n) {
    var s;
    try { s = typeof v === 'string' ? v : JSON.stringify(v); } catch (e) { s = String(v); }
    if (s == null) s = String(v);
    return s.length > (n || 200) ? s.slice(0, n || 200) + '...' : s;
  }
  function trim(s) { var t = String(s == null ? '' : s).replace(/\s+$/, ''); return brief(t, 160); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  // ── the funnel: one JSON string in, one JSON object out (spec §2) ────────
  // A future revision may accept an object literal instead of a JSON string;
  // both are accepted here so a signature change is a case failure with a
  // readable message rather than a stack trace in the middle of the suite.
  function invoke(ns, nsName, fnName, args) {
    if (!ns) throw new Error('sofuu.' + nsName + ' is not loaded in this build');
    var fn = ns[fnName];
    if (typeof fn !== 'function') throw new Error('sofuu.' + nsName + '.' + fnName + '() is missing in this build');
    var res;
    try {
      res = fn(JSON.stringify(args));
    } catch (eStr) {
      try { res = fn(args); }
      catch (eObj) { throw new Error(fnName + '() threw: ' + ((eObj && eObj.message) || eObj)); }
    }
    if (typeof res === 'string') {
      try { return JSON.parse(res); }
      catch (eParse) { throw new Error(fnName + '() returned non-JSON text: ' + brief(res, 120)); }
    }
    if (res && typeof res === 'object') return res;
    throw new Error(fnName + '() returned ' + typeof res + ' (expected a JSON object)');
  }
  function append(args) { return invoke(E, 'events', 'append', args); }
  function readLog(args) { return invoke(E, 'events', 'read', args); }
  function obs(fnName, args) { return invoke(O, 'observability', fnName, args); }
  function errorCode(res) { return (res && res.code) || (res && res.error && res.error.code) || ''; }
  function eventsOf(rd) { return (rd && rd.events) || []; }
  function assertNoErrors(res) {
    var errs = (res && res.errors) || [];
    assert(errs.length === 0, 'append() reported errors[]: ' + brief(errs));
  }

  // ── filesystem assertions (raw bytes, not the read() view) ──────────────
  async function readRaw(p) { return String(await FS.readFile(p)); }
  async function exists(p) { return !!(await FS.exists(p)); }
  function lines(text) { return String(text).split('\n').filter(function (l) { return l.length > 0; }); }
  function parseLines(text) {
    var out = { events: [], bad: [] };
    lines(text).forEach(function (l, i) {
      try { out.events.push(JSON.parse(l)); } catch (e) { out.bad.push('line ' + (i + 1) + ': ' + brief(l, 80)); }
    });
    return out;
  }

  // ── the run directory this suite owns (spec §1) ─────────────────────────
  var STAMP = Date.now().toString(36) + '-' + ((typeof process !== 'undefined' && process.pid) || Math.floor(Math.random() * 1e6));
  var TMP = '/tmp/sofuu-observability-stress-' + STAMP;
  var KEEP = !!(typeof process !== 'undefined' && process.env && process.env.SOFUU_OBS_KEEP === '1');
  var RUNS = 0;
  function runId(tag) { RUNS++; return 'obs-' + tag + '-' + STAMP + '-' + RUNS; }
  function runPath(id) { return TMP + '/.sofuu/logs/agent-' + id + '.jsonl'; }
  function sessionPath(id) { return TMP + '/.sofuu/sessions/' + id + '/events.ndjson'; }
  function crashPath(id) { return TMP + '/.sofuu/crashes/' + id + '.json'; }

  function ev(sid, kind, turn, payload) {
    return {
      kind: kind,
      schema_version: 1,
      sid: sid,
      turn_id: turn || 't-1',
      ts: new Date().toISOString(),
      surface: 'js',
      payload: payload || {}
    };
  }
  function checkpoint(sid, step, task) {
    return ev(sid, 'checkpoint', 't-1', {
      step: step,
      messages: [{ role: 'user', content: task || 'stress' }],
      task: task || 'stress',
      model: 'local',
      provider: 'local',
      tokens: step * 10,
      last_tool: 'read_file'
    });
  }
  function syncedFlag(res) {
    if (!res || typeof res.synced !== 'boolean') {
      throw new Error('append() result carries no boolean synced flag (spec §2 durability signal); keys=' + Object.keys(res || {}).join(','));
    }
    return res.synced;
  }

  // ── child-process plumbing (kill -9 and concurrency are cross-process) ──
  async function resolveBinary() {
    var argv = (typeof process !== 'undefined' && process.argv) || [];
    var cand = String(argv[0] || '');
    if (!cand) throw new Error('process.argv[0] is empty; cannot locate the sofuu binary');
    try { return String(await FS.realpath(cand)); } catch (e) {}
    if (cand.indexOf('/') >= 0) return cand; // a relative path resolves against the inherited cwd
    throw new Error('cannot resolve the sofuu binary from argv[0]=' + cand);
  }
  function spawnCapture(cmd, args, cwd) {
    return new Promise(function (resolve, reject) {
      var stdout = '', stderr = '';
      var opts = {
        command: cmd,
        args: args,
        onStdout: function (c) { if (stdout.length < 65536) stdout += String(c); },
        onStderr: function (c) { if (stderr.length < 65536) stderr += String(c); },
        onExit: function (code) { resolve({ code: code, stdout: stdout, stderr: stderr }); }
      };
      if (cwd) opts.cwd = cwd;
      try { sofuu.spawn(opts); }
      catch (e) { reject(new Error('spawn ' + cmd + ' failed: ' + ((e && e.message) || e))); }
    });
  }

  async function setup() {
    await FS.mkdir(TMP + '/.sofuu/logs', { recursive: true });
  }
  async function cleanup() {
    if (KEEP) { console.log('  kept: ' + TMP); return; }
    try { await sofuu.exec('/bin/rm', ['-rf', TMP]); } catch (e) {}
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 1 — append -> read round-trip with the full envelope (spec §0)
  // ═════════════════════════════════════════════════════════════════════════
  async function caseRoundtrip() {
    console.log('\n=== Cat 1: envelope round-trip ===');
    await check('A1 append -> read round-trip preserves the full envelope', async function () {
      var run = runId('roundtrip');
      var p = runPath(run);
      var env = ev(run, 'run_start', 't-1', { task: 'roundtrip', n: 1 });
      env.seq = 1;
      var res = append({ project: TMP, sid: run, path: p, events: [env] });
      assert(res && res.ok === true, 'append() did not report ok:true: ' + brief(res));
      eq(res.written, 1, 'written');
      assertNoErrors(res);
      eq(res.path, p, 'append() must echo the file it wrote');
      assert(await exists(p), 'append() did not create ' + p);

      var rd = readLog({ project: TMP, sid: run, path: p });
      assert(rd && rd.ok === true, 'read() did not report ok:true: ' + brief(rd));
      var got = eventsOf(rd)[0];
      assert(got, 'read() returned no events for a log with one line');
      ['kind', 'schema_version', 'sid', 'turn_id', 'seq', 'ts', 'surface', 'payload'].forEach(function (f) {
        assert(got[f] !== undefined && got[f] !== null, 'round-trip lost envelope field ' + f + ': ' + brief(got));
      });
      eq(got.kind, 'run_start', 'kind');
      eq(got.schema_version, 1, 'schema_version');
      eq(got.sid, run, 'sid');
      eq(got.turn_id, 't-1', 'turn_id');
      eq(got.seq, 1, 'seq');
      eq(got.surface, 'js', 'surface');
      eq(got.payload && got.payload.task, 'roundtrip', 'payload.task');
      if (rd.invalid !== undefined) eq(rd.invalid, 0, 'invalid lines');

      var raw = parseLines(await readRaw(p));
      eq(raw.events.length, 1, 'raw JSONL lines');
      eq(raw.bad.length, 0, 'raw JSONL parse failures: ' + brief(raw.bad));
      return 'bytes=' + (await readRaw(p)).length;
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 2 — seq auto-assignment is monotonic; regressions are refused
  // ═════════════════════════════════════════════════════════════════════════
  async function caseSeq() {
    console.log('\n=== Cat 2: seq total order ===');
    await check('A2 seq is auto-assigned monotonically and a regression lands in errors[]', async function () {
      var run = runId('seq');
      var p = runPath(run);
      var res = append({
        project: TMP, sid: run, path: p,
        events: [
          ev(run, 'delta', 't-1', { text: 'a' }),
          ev(run, 'delta', 't-1', { text: 'b' }),
          ev(run, 'delta', 't-1', { text: 'c' })
        ]
      });
      assert(res && res.ok === true, 'append() did not report ok:true: ' + brief(res));
      eq(res.written, 3, 'written');
      var seqs = eventsOf(readLog({ project: TMP, sid: run, path: p })).map(function (e) { return e.seq; });
      eq(seqs.length, 3, 'events in the log');
      assert(seqs[0] < seqs[1] && seqs[1] < seqs[2],
        'auto-assigned seq is not strictly increasing: ' + JSON.stringify(seqs));
      eq(res.seq_end, seqs[2], 'seq_end must equal the last assigned seq');

      // An explicit seq at or below the high-water mark is a regression: it
      // must be refused and NAMED in errors[], never silently renumbered.
      var stale = ev(run, 'delta', 't-1', { text: 'stale' });
      stale.seq = seqs[1];
      var reg = append({ project: TMP, sid: run, path: p, events: [stale] });
      eq(reg.ok, false, 'a seq regression must not report ok:true');
      eq(reg.written, 0, 'a seq regression must not be written');
      assert(reg.errors && reg.errors.length >= 1, 'a seq regression must be reported in errors[]');
      assert(JSON.stringify(reg.errors).indexOf('seq') >= 0,
        'errors[] must name the seq regression: ' + brief(reg.errors));
      eq(eventsOf(readLog({ project: TMP, sid: run, path: p })).length, 3,
        'the log must still hold exactly the accepted events');
      return 'seqs=' + JSON.stringify(seqs) + ' rejected=' + brief(reg.errors, 90);
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 3 — run_id is an alias for sid (spec §0)
  // ═════════════════════════════════════════════════════════════════════════
  async function caseRunIdAlias() {
    console.log('\n=== Cat 3: run_id alias ===');
    await check('A3 an event carrying run_id and no sid validates and is normalised', async function () {
      var run = runId('alias');
      var p = runPath(run);
      // Alias shape: run_id only — no sid anywhere in the event.
      var e = {
        kind: 'checkpoint',
        schema_version: 1,
        run_id: run,
        turn_id: 't-1',
        ts: new Date().toISOString(),
        surface: 'js',
        payload: { step: 1, messages: [], task: 'alias', model: 'local', provider: 'local', tokens: 0, last_tool: 'none' }
      };
      var res = append({ project: TMP, sid: run, path: p, events: [e] });
      assert(res && res.ok === true, 'an event with run_id and no sid was rejected: ' + brief(res));
      eq(res.written, 1, 'written');
      assertNoErrors(res);

      var got = eventsOf(readLog({ project: TMP, sid: run, path: p }))[0];
      assert(got, 'the run_id event was not stored');
      eq(got.sid, run, 'append() must normalise run_id into sid (spec §0)');
      eq(got.payload && got.payload.step, 1, 'payload.step');
      return 'normalised sid=' + got.sid;
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 4 — redaction: secrets never reach the bytes on disk
  // ═════════════════════════════════════════════════════════════════════════
  async function caseRedaction() {
    console.log('\n=== Cat 4: redaction at the write choke point ===');
    await check('A4 apiKey/secret/password/authorization/token never appear on disk', async function () {
      var run = runId('redact');
      var p = runPath(run);
      var secrets = {
        apiKey: 'sk-live-' + STAMP + '-aaa',
        secret: 'client-secret-' + STAMP + '-bbb',
        password: 'hunter2-' + STAMP + '-ccc',
        authorization: 'Bearer ' + STAMP + '-ddd',
        token: 'tok-' + STAMP + '-eee'
      };
      var payload = { name: 'http_post', keep: 'keep-me-' + STAMP };
      Object.keys(secrets).forEach(function (k) { payload[k] = secrets[k]; });
      payload.nested = { api_key: 'nested-' + STAMP + '-fff', note: 'nested-keep' };

      var res = append({ project: TMP, sid: run, path: p, events: [ev(run, 'tool_call', 't-1', payload)] });
      assert(res && res.ok === true, 'append() rejected the tool_call event: ' + brief(res));
      eq(res.written, 1, 'written');

      var raw = await readRaw(p);
      Object.keys(secrets).forEach(function (k) {
        assert(raw.indexOf(secrets[k]) < 0, 'the value of ' + k + ' reached the log on disk: ' + brief(secrets[k], 40));
      });
      assert(raw.indexOf('nested-' + STAMP) < 0, 'a nested secret (payload.nested.api_key) reached the log on disk');
      assert(raw.indexOf('[REDACTED]') >= 0, 'no [REDACTED] marker in the log, so nothing was redacted');
      assert(raw.indexOf('keep-me-' + STAMP) >= 0, 'redaction destroyed a non-secret payload field');
      assert(raw.indexOf('nested-keep') >= 0, 'redaction destroyed a non-secret nested field');
      return 'markers=' + (raw.match(/\[REDACTED\]/g) || []).length;
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 5 — explicit path override (spec §2 path)
  // ═════════════════════════════════════════════════════════════════════════
  async function casePathOverride() {
    console.log('\n=== Cat 5: path override ===');
    await check('A5 events land at the requested .sofuu/logs/agent-<run>.jsonl', async function () {
      var run = runId('path');
      var want = runPath(run);
      var res = append({ project: TMP, sid: run, path: want, events: [ev(run, 'run_start', 't-1', { task: 'path' })] });
      assert(res && res.ok === true, 'append() with an explicit path failed: ' + brief(res));
      eq(res.path, want, 'append() must write the requested path');
      assert(await exists(want), 'no file at the requested path ' + want);
      assert(!(await exists(sessionPath(run))),
        'the event also landed in .sofuu/sessions/<sid>/events.ndjson; the path override was ignored');

      var raw = parseLines(await readRaw(want));
      eq(raw.events.length, 1, 'lines at the requested path');
      eq(raw.bad.length, 0, 'parse failures at the requested path');
      eq(raw.events[0].sid, run, 'sid at the requested path');
      return res.path.replace(TMP, '<tmp>');
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 6 — durability signal: checkpoint batches fsync, deltas do not
  // ═════════════════════════════════════════════════════════════════════════
  async function caseDurability() {
    console.log('\n=== Cat 6: durability (fsync on checkpoint) ===');
    await check('A6 a checkpoint batch reports synced:true and a deltas-only batch synced:false', async function () {
      var deltaRun = runId('durable-deltas');
      var pDeltas = runPath(deltaRun);
      var plain = append({
        project: TMP, sid: deltaRun, path: pDeltas,
        events: [ev(deltaRun, 'delta', 't-1', { text: 'x' }), ev(deltaRun, 'delta', 't-1', { text: 'y' })]
      });
      assert(plain && plain.ok === true, 'the deltas-only append failed: ' + brief(plain));
      eq(syncedFlag(plain), false, 'a deltas-only batch must not pay for fsync (synced:false)');

      var ckRun = runId('durable-checkpoint');
      var pCk = runPath(ckRun);
      var durable = append({ project: TMP, sid: ckRun, path: pCk, events: [checkpoint(ckRun, 1, 'durable')] });
      assert(durable && durable.ok === true, 'the checkpoint append failed: ' + brief(durable));
      eq(syncedFlag(durable), true, 'a batch containing a checkpoint must fsync automatically and report synced:true');

      // The spec's explicit durable flag is the caller's override.
      var forced = append({ project: TMP, sid: deltaRun, path: pDeltas, durable: true, events: [ev(deltaRun, 'delta', 't-1', { text: 'z' })] });
      assert(forced && forced.ok === true, 'the durable:true append failed: ' + brief(forced));
      eq(syncedFlag(forced), true, 'durable:true must report synced:true');
      return 'deltas=false checkpoint=true durable=true';
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 7 — crash post-mortem card + crashCard() newest-wins
  // ═════════════════════════════════════════════════════════════════════════
  async function casePostmortem() {
    console.log('\n=== Cat 7: crash post-mortem cards ===');
    await check('A7 postmortem() writes the plan card shape and crashCard() returns the newest', async function () {
      var run = runId('crash');
      var pm = obs('postmortem', {
        project: TMP,
        run_id: run,
        step: 14,
        error: {
          code: 'E_SCHEMA',
          message: "Tool 'transfer_funds' failed argument validation",
          details: { missing_fields: ['recipient_account'], received_payload: { amount: 100 } },
          stack: 'Error: E_SCHEMA\n  at runTool (plugins.js:142)'
        },
        context: {
          model: 'llama3.2',
          provider: 'local',
          context_tokens: 4210,
          last_user_message: 'Send $100 to Alice',
          last_tool: 'transfer_funds'
        },
        recovery_hint: 'Run sofuu resume ' + run + ' after updating the tool schema.'
      });
      assert(pm && pm.ok === true, 'postmortem() did not report ok:true: ' + brief(pm));
      assert(typeof pm.crash_id === 'string' && pm.crash_id.length > 0, 'postmortem() returned no crash_id');
      var tail = 'crashes/' + pm.crash_id + '.json';
      var pmPath = String(pm.path || '');
      assert(pmPath.length >= tail.length && pmPath.indexOf(tail) === pmPath.length - tail.length,
        'postmortem() path must end with .sofuu/' + tail + '; got ' + brief(pmPath));

      var onDiskPath = crashPath(pm.crash_id);
      assert(await exists(onDiskPath), 'no crash card at ' + onDiskPath);
      var onDisk = JSON.parse(await readRaw(onDiskPath));

      var card = pm.card || onDisk;
      ['crash_id', 'run_id', 'timestamp', 'failed_step', 'error', 'context_snapshot', 'recovery_hint'].forEach(function (f) {
        assert(card[f] !== undefined && card[f] !== null, 'the crash card is missing the plan field ' + f);
      });
      eq(card.crash_id, pm.crash_id, 'card.crash_id');
      eq(card.run_id, run, 'card.run_id');
      eq(card.failed_step, 14, 'card.failed_step');
      eq(card.error.code, 'E_SCHEMA', 'card.error.code');
      var ctx = card.context_snapshot || {};
      ['active_model', 'context_tokens_used', 'last_user_message', 'last_tool_in_flight'].forEach(function (f) {
        assert(ctx[f] !== undefined && ctx[f] !== null, 'context_snapshot is missing ' + f + ' (OBSERVABILITY_PLAN Component 3)');
      });
      eq(ctx.active_model, 'llama3.2', 'context_snapshot.active_model');
      eq(ctx.context_tokens_used, 4210, 'context_snapshot.context_tokens_used');
      eq(onDisk.crash_id, pm.crash_id, 'the card on disk is not the card postmortem() returned');
      eq(onDisk.error.code, 'E_SCHEMA', 'the card on disk lost error.code');

      // Newest-wins: a second card for the same run must be the one returned.
      // The sleep keeps the two ids distinct at second granularity, so the
      // assertion does not depend on the id format the lane picks.
      await sleep(1100);
      var pm2 = obs('postmortem', {
        project: TMP,
        run_id: run,
        step: 15,
        error: { code: 'E_TIMEOUT', message: 'upstream timed out after 30s' },
        context: { model: 'llama3.2', provider: 'local', context_tokens: 4300, last_user_message: 'Send $100 to Alice', last_tool: 'transfer_funds' },
        recovery_hint: 'Run sofuu resume ' + run + '.'
      });
      assert(pm2 && pm2.ok === true, 'the second postmortem() failed: ' + brief(pm2));
      assert(pm2.crash_id !== pm.crash_id, 'postmortem() reused a crash_id for a second crash (cards overwrite)');

      var cc = obs('crashCard', { project: TMP, run_id: run });
      assert(cc && cc.ok === true && cc.card, 'crashCard({run_id}) returned no card: ' + brief(cc));
      eq(cc.card.crash_id, pm2.crash_id, 'crashCard() must return the NEWEST card for the run');
      var newest = obs('crashCard', { project: TMP });
      assert(newest && newest.ok === true && newest.card, 'crashCard({project}) returned no card: ' + brief(newest));
      eq(newest.card.crash_id, pm2.crash_id, 'crashCard() without a run must return the newest card overall');
      return 'cards=2 newest=' + pm2.crash_id;
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 8 — lastCheckpoint survives a torn trailing line
  // ═════════════════════════════════════════════════════════════════════════
  async function caseLastCheckpoint() {
    console.log('\n=== Cat 8: lastCheckpoint reverse scan ===');
    await check('A8 lastCheckpoint returns the NEWEST checkpoint even after a torn line', async function () {
      var run = runId('checkpoint');
      var p = runPath(run);
      var res = append({
        project: TMP, sid: run, path: p,
        events: [
          ev(run, 'run_start', 't-1', { task: 'checkpoint' }),
          checkpoint(run, 1, 'step one'),
          ev(run, 'delta', 't-1', { text: 'between' }),
          checkpoint(run, 2, 'step two'),
          checkpoint(run, 3, 'step three')
        ]
      });
      assert(res && res.ok === true, 'seeding the checkpoint log failed: ' + brief(res));
      eq(res.written, 5, 'written');

      var lc = obs('lastCheckpoint', { project: TMP, run_id: run });
      assert(lc && lc.ok === true, 'lastCheckpoint() did not report ok:true: ' + brief(lc));
      eq(lc.found, true, 'found');
      eq(lc.step, 3, 'lastCheckpoint().step must be the NEWEST checkpoint step, not the first');
      assert(typeof lc.seq === 'number' && lc.seq > 1, 'lastCheckpoint().seq must be the checkpoint event seq: ' + brief(lc));
      assert(lc.checkpoint && typeof lc.checkpoint === 'object', 'lastCheckpoint() returned no checkpoint payload');
      var ckPayload = lc.checkpoint.payload || lc.checkpoint;
      eq(ckPayload.step, 3, 'the returned checkpoint payload is not the newest one');

      // A power cut leaves a half-written line behind. The reverse scan must
      // skip it rather than hide the last good checkpoint.
      await FS.appendFile(p, '{"kind":"checkpoint","step":99,"sid":"' + run + '"\n');
      await FS.appendFile(p, 'not json at all\n');
      var lc2 = obs('lastCheckpoint', { project: TMP, run_id: run });
      assert(lc2 && lc2.ok === true, 'lastCheckpoint() failed on a log with a torn line: ' + brief(lc2));
      eq(lc2.found, true, 'found must stay true when the trailing line is torn');
      eq(lc2.step, 3, 'a torn trailing line must not hide the newest valid checkpoint');
      return 'newest step=' + lc2.step + ' seq=' + lc2.seq;
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 9 — runs() enumerates both kinds and derives the status
  // ═════════════════════════════════════════════════════════════════════════
  async function caseRuns() {
    console.log('\n=== Cat 9: runs() enumeration + status derivation ===');
    await check('A9 runs() enumerates agent and session logs with derived status', async function () {
      var okRun = runId('runs-ok');
      var haltRun = runId('runs-halted');
      var retryRun = runId('runs-retrying');
      var liveRun = runId('runs-running');
      var sessId = runId('runs-session');

      var seeded = append({
        project: TMP, sid: okRun, path: runPath(okRun),
        events: [
          ev(okRun, 'run_start', 't-1', { task: 'runs' }),
          checkpoint(okRun, 1, 'runs'),
          ev(okRun, 'run_complete', 't-1', { steps: 1 })
        ]
      });
      assert(seeded && seeded.ok === true, 'seeding the OK run failed: ' + brief(seeded));
      seeded = append({
        project: TMP, sid: haltRun, path: runPath(haltRun),
        events: [
          ev(haltRun, 'run_start', 't-1', { task: 'runs' }),
          ev(haltRun, 'fatal_error', 't-1', { code: 'E_INTERNAL', message: 'boom' }),
          ev(haltRun, 'halt', 't-1', { reason: 'fatal_error' })
        ]
      });
      assert(seeded && seeded.ok === true, 'seeding the halted run failed: ' + brief(seeded));
      seeded = append({
        project: TMP, sid: retryRun, path: runPath(retryRun),
        events: [
          ev(retryRun, 'run_start', 't-1', { task: 'runs' }),
          ev(retryRun, 'retry', 't-1', { attempt: 1, delay_ms: 250, code: 'E_TIMEOUT' })
        ]
      });
      assert(seeded && seeded.ok === true, 'seeding the retrying run failed: ' + brief(seeded));
      seeded = append({ project: TMP, sid: liveRun, path: runPath(liveRun), events: [ev(liveRun, 'run_start', 't-1', { task: 'runs' })] });
      assert(seeded && seeded.ok === true, 'seeding the running run failed: ' + brief(seeded));
      // No path: this one belongs to the unchanged chat-session layout (spec §1).
      seeded = append({
        project: TMP, sid: sessId,
        events: [ev(sessId, 'delta', 't-1', { text: 'session' }), ev(sessId, 'stop', 't-1', { reason: 'done' })]
      });
      assert(seeded && seeded.ok === true, 'seeding the session log failed: ' + brief(seeded));

      var rs = obs('runs', { project: TMP });
      assert(rs && rs.ok === true, 'runs() did not report ok:true: ' + brief(rs));
      assert(Object.prototype.toString.call(rs.runs) === '[object Array]', 'runs() returned no runs array');

      var byId = {};
      rs.runs.forEach(function (r) { byId[r.run_id] = r; });
      function expect(id, kind, status, label) {
        var r = byId[id];
        assert(r, label + ' run ' + id + ' is not enumerated by runs()');
        eq(r.kind, kind, label + ' run kind');
        eq(r.status, status, label + ' run status');
        assert(typeof r.path === 'string' && r.path.length > 0, label + ' run has no path');
        assert(typeof r.bytes === 'number' && r.bytes > 0, label + ' run has no byte size');
        // The spec does not type mtime: epoch seconds and an ISO-8601 stamp
        // are both faithful renderings, so accept either.
        var hasMtime = (typeof r.mtime === 'number' && r.mtime > 0)
          || (typeof r.mtime === 'string' && r.mtime.length > 0);
        assert(hasMtime, label + ' run has no mtime');
        assert(typeof r.steps === 'number', label + ' run has no step count');
        return r;
      }
      expect(okRun, 'agent', 'OK', 'completing');
      expect(haltRun, 'agent', 'HALTED', 'halted');
      expect(retryRun, 'agent', 'RETRYING', 'retrying');
      expect(liveRun, 'agent', 'RUNNING', 'live');

      var sessions = rs.runs.filter(function (r) { return r.kind === 'session'; });
      assert(sessions.length >= 1, 'runs() does not enumerate session logs as kind session');
      assert(sessions[0].path.indexOf('.sofuu/sessions/') >= 0,
        'a session run path must live under .sofuu/sessions/: ' + sessions[0].path);
      var kinds = {};
      rs.runs.forEach(function (r) { kinds[r.kind] = (kinds[r.kind] || 0) + 1; });
      return 'runs=' + rs.runs.length + ' kinds=' + JSON.stringify(kinds);
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 10 — kill -9: the checkpoint written before the signal must survive
  // ═════════════════════════════════════════════════════════════════════════
  function kill9ChildSource(id, project, logPath, sentinel) {
    return [
      '// generated by scripts/stress/observability_stress.js — kill -9 probe.',
      '// Append one checkpoint (fsync is automatic for that kind), mark it, then',
      '// block forever so the parent can SIGKILL a child that never ran teardown.',
      '(async function () {',
      '  var res = sofuu.events.append(JSON.stringify({',
      '    project: ' + JSON.stringify(project) + ',',
      '    sid: ' + JSON.stringify(id) + ',',
      '    path: ' + JSON.stringify(logPath) + ',',
      '    durable: true,',
      '    events: [{ kind: "checkpoint", schema_version: 1, sid: ' + JSON.stringify(id) + ', turn_id: "t-1",',
      '      ts: new Date().toISOString(), surface: "js", payload: { step: 7, task: "kill9 probe",',
      '      messages: [{ role: "user", content: "kill9" }], model: "local", provider: "local", tokens: 7, last_tool: "none" } }]',
      '  }));',
      '  await sofuu.fs.writeFile(' + JSON.stringify(sentinel) + ', String(res));',
      '  for (;;) {}',
      '})();',
      ''
    ].join('\n');
  }

  async function caseKill9() {
    console.log('\n=== Cat 10: kill -9 durability ===');
    await check('A10 a checkpoint survives SIGKILL of the writing child', async function () {
      if (!E || typeof E.append !== 'function') throw new Error('sofuu.events.append() is missing, so there is nothing to prove');
      var bin = await resolveBinary();
      var run = runId('kill9');
      var p = runPath(run);
      var child = TMP + '/kill9_child.js';
      var sentinel = TMP + '/kill9_child.sentinel';
      var childLog = TMP + '/kill9_child.log';
      await FS.writeFile(child, kill9ChildSource(run, TMP, p, sentinel));

      // A shell driver is the only way to deliver a real SIGKILL: sofuu.spawn's
      // kill() sends SIGTERM, which a well-behaved child could handle.
      var script = [
        '"$1" run "$2" >"$4" 2>&1 &',
        'pid=$!',
        'i=0',
        'while [ ! -f "$3" ] && [ "$i" -lt 150 ]; do sleep 0.1; i=$((i+1)); done',
        'sleep 0.2',
        'kill -9 "$pid" 2>/dev/null',
        'wait "$pid" 2>/dev/null',
        'echo "KILLSTATUS=$?"'
      ].join('\n');
      var out = await spawnCapture('/bin/sh', ['-c', script, 'sofuu-kill9', bin, child, sentinel, childLog]);
      var childText = (await exists(childLog)) ? await readRaw(childLog) : '';

      eq(/KILLSTATUS=137/.test(out.stdout), true,
        'the child did not die by SIGKILL (driver stdout: ' + trim(out.stdout) + '; child log: ' + trim(childText) + ')');
      assert(await exists(sentinel),
        'the child never reported its checkpoint before the SIGKILL (child log: ' + trim(childText) + ')');
      var sent = JSON.parse(await readRaw(sentinel));
      assert(sent && sent.ok === true, 'the child append() did not report ok:true: ' + brief(sent));
      eq(sent.written, 1, 'the child wrote no event');

      // The child is dead and could not have flushed anything on the way out:
      // whatever is on disk now is exactly what the checkpoint fsync bought.
      assert(await exists(p), 'no log at ' + p + ' after the SIGKILL');
      var raw = parseLines(await readRaw(p));
      eq(raw.bad.length, 0, 'corrupted lines after SIGKILL: ' + brief(raw.bad));
      var cks = raw.events.filter(function (e) { return e.kind === 'checkpoint'; });
      assert(cks.length === 1, 'expected exactly one checkpoint after the SIGKILL, found ' + cks.length);
      eq(cks[0].payload && cks[0].payload.step, 7, 'the surviving checkpoint payload');
      return 'killed, checkpoint step=7 readable (' + (await readRaw(p)).length + ' bytes)';
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 11 — four concurrent appenders, one log
  // ═════════════════════════════════════════════════════════════════════════
  function concurrentWriterSource() {
    return [
      '// generated by scripts/stress/observability_stress.js — one appender.',
      '// delta is one of the base ten kinds, so this case isolates append',
      '// concurrency from whether the new agent kinds are registered yet.',
      '(function () {',
      '  var a = process.argv;',
      '  var path = a[3], sid = a[4], tag = a[5], n = Number(a[6]), project = a[7];',
      '  var ok = 0, bad = 0, lastErr = "";',
      '  for (var i = 0; i < n; i++) {',
      '    var r = JSON.parse(sofuu.events.append(JSON.stringify({',
      '      project: project, sid: sid, path: path,',
      '      events: [{ kind: "delta", schema_version: 1, sid: sid, turn_id: "turn-" + tag,',
      '        ts: new Date().toISOString(), surface: "js",',
      '        payload: { text: tag + ":" + i, tag: tag, i: i } }]',
      '    })));',
      '    if (r && r.ok) ok++;',
      '    else { bad++; lastErr = String(JSON.stringify((r && (r.errors || r.code)) || r)); }',
      '  }',
      '  console.log("WROTE " + ok + " BAD " + bad + (lastErr ? " LAST " + lastErr.slice(0, 160) : ""));',
      '})();',
      ''
    ].join('\n');
  }

  // A11 proves the O_APPEND guarantee (complete lines, one event per line);
  // A11b proves the seq total order survives the same race. Two cases, because
  // they break for different reasons and are fixed in different places: a
  // non-atomic line write, and a high-water seq read without a cross-process
  // lock.
  var CONC = null;

  async function caseConcurrent() {
    console.log('\n=== Cat 11: concurrent appenders ===');
    await check('A11 four writers x 50 appends leave clean, line-valid JSONL', async function () {
      if (!E || typeof E.append !== 'function') throw new Error('sofuu.events.append() is missing, so there is nothing to append');
      var bin = await resolveBinary();
      var run = runId('concurrent');
      var p = runPath(run);
      var writer = TMP + '/concurrent_writer.js';
      await FS.writeFile(writer, concurrentWriterSource());

      var WRITERS = 4, PER = 50;
      var procs = [];
      for (var w = 0; w < WRITERS; w++) {
        procs.push(spawnCapture(bin, ['run', writer, p, run, 'w' + w, String(PER), TMP]));
      }
      var outs = await Promise.all(procs);
      for (var i = 0; i < outs.length; i++) {
        eq(outs[i].code, 0, 'writer w' + i + ' exited non-zero (stderr: ' + trim(outs[i].stderr) + ')');
        eq(trim(outs[i].stdout), 'WROTE ' + PER + ' BAD 0',
          'writer w' + i + ' did not report ' + PER + ' accepted appends');
      }

      assert(await exists(p), 'the concurrent writers produced no log at ' + p);
      var ls = lines(await readRaw(p));
      var broken = [], events = [];
      ls.forEach(function (l, idx) {
        try { events.push(JSON.parse(l)); }
        catch (e) { broken.push('line ' + (idx + 1) + ': ' + brief(l, 120)); }
      });
      // Two events glued onto one line is exactly what a non-atomic line write
      // looks like: the JSON and its newline are two separate appends, so
      // another process can slip its own record in between them.
      CONC = { path: p, lines: ls.length, events: events, broken: broken };
      assert(broken.length === 0,
        broken.length + ' interleaved/corrupted line(s) out of ' + ls.length +
        ' (first: ' + brief(broken[0], 220) + ')');
      eq(ls.length, WRITERS * PER, 'lines in the log (torn or lost appends)');
      events.forEach(function (e, idx) {
        eq(e.sid, run, 'line ' + (idx + 1) + ' sid');
        eq(e.kind, 'delta', 'line ' + (idx + 1) + ' kind');
        assert(e.payload && typeof e.payload.tag === 'string', 'line ' + (idx + 1) + ' payload.tag');
      });
      return 'lines=' + ls.length + ' writers=' + WRITERS;
    });
  }

  async function caseConcurrentSeq() {
    await check('A11b concurrent appends keep seq strictly increasing and collision-free', async function () {
      assert(CONC, 'A11 did not run, so there is no concurrent log to inspect');
      assert(CONC.events.length > 0, 'the concurrent log held no parseable events (see A11)');
      var seqs = CONC.events.map(function (e) { return e.seq; });
      var seen = {}, dupes = [];
      seqs.forEach(function (s) {
        if (typeof s !== 'number') throw new Error('an event has no numeric seq: ' + JSON.stringify(s));
        if (seen[s]) dupes.push(s);
        seen[s] = 1;
      });
      assert(dupes.length === 0,
        'seq collided across concurrent appenders (' + dupes.length + ' duplicate value(s), e.g. ' +
        JSON.stringify(dupes.slice(0, 5)) + '); first seqs in file order: ' + JSON.stringify(seqs.slice(0, 8)) +
        '. append() reads the high-water seq without serialising across processes.');
      for (var j = 1; j < seqs.length; j++) {
        assert(seqs[j] > seqs[j - 1],
          'file-order seq is not strictly increasing at line ' + (j + 1) + ': ' +
          JSON.stringify(seqs.slice(Math.max(0, j - 3), j + 3)) + ' (first seqs: ' + JSON.stringify(seqs.slice(0, 8)) + ')');
      }
      return 'seq=' + seqs[0] + '..' + seqs[seqs.length - 1] + ' over ' + seqs.length + ' events';
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 12 — oversized batch is a typed refusal, not a silent drop
  // ═════════════════════════════════════════════════════════════════════════
  async function caseBackpressure() {
    console.log('\n=== Cat 12: backpressure ===');
    await check('A12 a >1024 event batch returns E_BACKPRESSURE and writes nothing', async function () {
      var run = runId('backpressure');
      var p = runPath(run);
      var big = [];
      for (var i = 0; i < 1025; i++) big.push(ev(run, 'delta', 't-1', { text: 'n' + i }));
      var res = append({ project: TMP, sid: run, path: p, events: big });
      eq(res.ok, false, 'an oversized batch must not report ok:true');
      eq(errorCode(res), 'E_BACKPRESSURE', 'oversized batch must be a typed E_BACKPRESSURE (spec §0 taxonomy)');
      assert(!res.written, 'an oversized batch must not write a prefix (written=' + res.written + ')');
      if (await exists(p)) {
        eq(lines(await readRaw(p)).length, 0, 'lines written by the rejected batch');
      }

      // 1024 is the documented bound, so exactly-at-the-bound is accepted.
      var at = [];
      for (var k = 0; k < 1024; k++) at.push(ev(run, 'delta', 't-1', { text: 'b' + k }));
      var res2 = append({ project: TMP, sid: run, path: p, events: at });
      assert(res2 && res2.ok === true,
        'a 1024-event batch is at the bound and must be accepted, got: ' + brief(res2));
      eq(res2.written, 1024, 'events written at the bound');
      return 'rejected 1025=' + errorCode(res) + ' accepted 1024';
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Case 13 — malformed input
  // ═════════════════════════════════════════════════════════════════════════
  async function caseMalformed() {
    console.log('\n=== Cat 13: malformed input ===');
    await check('A13 append without sid is E_SCHEMA; an unregistered kind lands in errors[]', async function () {
      // No sid AND no path: nothing identifies the log this could belong to.
      var noSid = append({ project: TMP, events: [ev('ignored', 'delta', 't-1', { text: 'x' })] });
      eq(noSid.ok, false, 'append() without a sid must not report ok:true');
      eq(errorCode(noSid), 'E_SCHEMA', 'a missing sid must be E_SCHEMA (spec §0 taxonomy)');

      var run = runId('malformed');
      var p = runPath(run);
      var badKind = append({
        project: TMP, sid: run, path: p,
        events: [{
          kind: 'totally_made_up', schema_version: 1, sid: run, turn_id: 't-1',
          ts: new Date().toISOString(), surface: 'js', payload: {}
        }]
      });
      eq(badKind.ok, false, 'an unregistered kind must not report ok:true');
      eq(badKind.written, 0, 'an unregistered kind must not be written');
      assert(badKind.errors && badKind.errors.length >= 1, 'the rejected event was not listed in errors[]');
      var es = JSON.stringify(badKind.errors);
      assert(es.indexOf('totally_made_up') >= 0 || es.indexOf('unknown kind') >= 0 || es.indexOf('invalid') >= 0,
        'errors[] does not name the unregistered kind: ' + brief(es, 240));
      if (await exists(p)) eq(lines(await readRaw(p)).length, 0, 'lines written by the rejected kind');
      return 'E_SCHEMA + errors[]=' + badKind.errors.length;
    });
  }

  // ── receipt + summary (same shape the plugin suite emits) ───────────────
  function receipt() {
    var total = Math.round(Date.now() - R.startedAt);
    var out = {
      suite: 'observability_stress',
      project: TMP,
      cases: R.passed + R.failed + R.skipped,
      passed: R.passed,
      failed: R.failed,
      skipped: R.skipped,
      wall_ms: total,
      events_api: !!(E && typeof E.append === 'function'),
      observability_api: !!O,
      failures: R.cases.filter(function (c) { return !c.ok; }).map(function (c) { return c.name; })
    };
    console.log('\n=== SUMMARY ===');
    console.log('cases=' + out.cases + ' PASS=' + out.passed + ' FAIL=' + out.failed + ' SKIP=' + out.skipped + ' wall=' + total + 'ms');
    console.log('OBSERVABILITY_STRESS_RECEIPT ' + JSON.stringify(out));
    return out;
  }

  (async function main() {
    console.log('=== Sofuu Observability Stress Suite ===');
    console.log('surface: events=' + (E ? 'yes' : 'NO') + ' observability=' + (O ? 'yes' : 'NO') + ' project=' + TMP);
    try { await setup(); }
    catch (e) {
      console.error('FATAL: cannot create the scratch project ' + TMP + ': ' + ((e && e.message) || e));
      if (typeof process !== 'undefined' && process.exit) process.exit(2);
      return;
    }
    // A hung child must not hang the gate: bail out with a receipt instead.
    var watchdog = setTimeout(function () {
      console.error('FATAL: the suite exceeded its 180s wall budget');
      receipt();
      try { if (typeof process !== 'undefined' && process.exit) process.exit(2); } catch (eW) {}
    }, 180000);
    try {
      await caseRoundtrip();
      await caseSeq();
      await caseRunIdAlias();
      await caseRedaction();
      await casePathOverride();
      await caseDurability();
      await casePostmortem();
      await caseLastCheckpoint();
      await caseRuns();
      await caseKill9();
      await caseConcurrent();
      await caseConcurrentSeq();
      await caseBackpressure();
      await caseMalformed();
    } catch (e) {
      console.error('FATAL in suite: ' + ((e && e.stack) || e));
      R.failed++;
    }
    clearTimeout(watchdog);
    var out = receipt();
    await cleanup();
    try { if (typeof process !== 'undefined' && process.exit) process.exit(out.failed > 0 ? 1 : 0); } catch (eX) {}
  })();
})();
