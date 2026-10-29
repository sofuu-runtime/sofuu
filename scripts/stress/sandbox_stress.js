// scripts/stress/sandbox_stress.js — SANDBOX_UPGRADE_PLAN W6.2 adversarial suite.
//
// Twenty-two cases across the v2 attack surface: budget escapes, memory and
// stack bombs, catastrophic regex backtracking, host-global escape attempts,
// prototype pollution, TypeScript/JSX ingest, the callable bridge seam (sync,
// async, real I/O, throws, non-serializable results), console routing,
// presets, parked-run lifecycle, input validation and a run-count/RSS soak.
//
//   make sandbox-stress
//   ./sofuu run scripts/stress/sandbox_stress.js
//
// Every case is bounded (fuel, deadline, or a finite loop), so a hung engine
// shows up as a failure, never as a stuck terminal.

var pass = 0, fail = 0;
function ok(name, detail) { pass++; console.log('  PASS ' + name + (detail ? '  [' + detail + ']' : '')); }
function bad(name, detail) { fail++; console.log('  FAIL ' + name + '  [' + detail + ']'); }

async function check(name, fn) {
  try {
    var detail = await fn();
    if (detail === false) { bad(name, 'assertion returned false'); return; }
    ok(name, typeof detail === 'string' ? detail : '');
  } catch (e) {
    bad(name, (e && e.message) || String(e));
  }
}

function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

(async function main() {
  'use strict';
  console.log('=== Sandbox v2 adversarial stress (22 cases) ===');

  /* ── 1. Budget escapes ─────────────────────────────────────── */

  await check('A1 default budget stops an infinite loop', function () {
    var t = Date.now();
    var r = sofuu.sandbox.run('while(true){}');
    assert(!r.success, 'must fail');
    assert(r.interrupted === true, 'must be marked interrupted');
    assert(r.errorType === 'fuel' || r.errorType === 'timeout', 'type: ' + r.errorType);
    return (Date.now() - t) + 'ms, ' + r.errorType;
  });

  await check('A2 fuel:unlimited is stopped by the deadline', function () {
    var r = sofuu.sandbox.run('while(true){}', { fuel: 'unlimited', timeoutMs: 25 });
    assert(!r.success && r.errorType === 'timeout', 'got ' + r.errorType);
    assert(/E_TIMEOUT/.test(r.error), 'code in message');
    return r.errorType;
  });

  await check('A3 timeout:null is stopped by the meter', function () {
    var r = sofuu.sandbox.run('while(true){}', { fuel: 40000, timeoutMs: null });
    assert(!r.success && r.errorType === 'fuel', 'got ' + r.errorType);
    return 'fuel=' + r.fuelUsed;
  });

  await check('A4 a stated 5-minute budget is not clamped', function () {
    var r = sofuu.sandbox.run('1 + 1', { timeoutMs: 300000, fuel: 100000000 });
    assert(r.success && r.result === 2, JSON.stringify(r));
    return 'accepted 300000ms / 1e8 fuel';
  });

  await check('A5 unknown option values are refused', function () {
    var refused = 0;
    var cases = [
      function () { sofuu.sandbox.run('1', { timeoutMs: 'soon' }); },
      function () { sofuu.sandbox.run('1', { fuel: {} }); },
      function () { sofuu.sandbox.run('1', { preset: 'turbo' }); },
      function () { sofuu.sandbox.run('1', { console: 'loud' }); },
      function () { sofuu.sandbox.run('1', { ts: 'maybe' }); },
      function () { sofuu.sandbox.run('1', { maxMemoryMB: 'big' }); },
    ];
    for (var i = 0; i < cases.length; i++) {
      try { cases[i](); } catch (e) { refused++; }
    }
    assert(refused === cases.length, 'refused ' + refused + '/' + cases.length);
    return refused + '/' + cases.length + ' refused';
  });

  /* ── 2. Resource bombs ─────────────────────────────────────── */

  await check('B1 memory bomb -> E_OOM', function () {
    var r = sofuu.sandbox.run(
      'const a = []; while(true) a.push(new Uint8Array(262144));',
      { maxMemoryMB: 4, timeoutMs: null }
    );
    assert(!r.success && r.errorType === 'oom', 'got ' + r.errorType + ': ' + r.error);
    return r.errorType;
  });

  await check('B2 stack bomb -> E_STACK', function () {
    var r = sofuu.sandbox.run('function f(n){return f(n+1)} f(0);', { maxStackKB: 128 });
    assert(!r.success && r.errorType === 'stack', 'got ' + r.errorType);
    return r.errorType;
  });

  await check('B3 catastrophic regex -> deadline, session survives', function () {
    var sb = sofuu.sandbox.create();
    var r = sb.run('/(a+)+$/.test("a".repeat(26) + "b")', { timeoutMs: 80, fuel: null });
    assert(!r.success && r.errorType === 'timeout', 'got ' + r.errorType);
    var after = sb.run('/abc/.test("xabcx")');
    assert(after.success && after.result === true, 'realm usable after abort');
    sb.close();
    return r.errorType;
  });

  /* ── 3. Isolation ──────────────────────────────────────────── */

  await check('C1 no host globals are reachable', function () {
    var r = sofuu.sandbox.run(
      '({sofuu: typeof sofuu, process: typeof process, fetch: typeof fetch, ' +
      'require: typeof require, fs: typeof globalThis.fs, ' +
      'deno: typeof Deno, bun: typeof Bun, Buffer: typeof Buffer})'
    );
    assert(r.success, r.error || 'probe failed');
    var k = Object.keys(r.result);
    for (var i = 0; i < k.length; i++) {
      assert(r.result[k[i]] === 'undefined', k[i] + ' leaked: ' + r.result[k[i]]);
    }
    return k.length + ' globals absent';
  });

  await check('C2 constructor climbing stays in the realm', function () {
    var r = sofuu.sandbox.run(
      'const g = (function(){}).constructor("return this")(); ' +
      'typeof g.process + "," + typeof g.sofuu + "," + (g === globalThis)'
    );
    assert(r.success, r.error || 'probe failed');
    assert(r.result === 'undefined,undefined,true', 'got ' + r.result);
    return r.result;
  });

  await check('C3 sessions do not share globals', function () {
    var a = sofuu.sandbox.create();
    var b = sofuu.sandbox.create();
    a.run('globalThis.shared = 42;');
    var r = b.run('typeof globalThis.shared');
    assert(r.result === 'undefined', 'leaked: ' + r.result);
    a.close(); b.close();
    return 'isolated';
  });

  /* ── 4. Prototype pollution ────────────────────────────────── */

  await check('D1 freezePrototypes blocks pollution across turns', function () {
    var sb = sofuu.sandbox.create({ preset: 'strict' });
    sb.run('try { Object.prototype.evil = 1; } catch (e) {}');
    var r = sb.run('({p: ({}).evil === undefined, a: [].evil === undefined})');
    assert(r.result.p === true && r.result.a === true, JSON.stringify(r.result));
    var normal = sb.run('const a=[3,1,2]; a.sort(); a.join("-")');
    assert(normal.result === '1-2-3', 'ordinary code broken by freeze: ' + normal.result);
    sb.close();
    return 'frozen';
  });

  await check('D2 unfrozen pollution never reaches the host', function () {
    var r = sofuu.sandbox.run('Object.prototype.evilHost = 1; ({}).evilHost');
    assert(r.success && r.result === 1, 'inside realm it is visible');
    assert(({}).evilHost === undefined, 'pollution leaked into the host realm');
    return 'contained';
  });

  /* ── 5. TypeScript / JSX ingest ────────────────────────────── */

  await check('E1 TypeScript constructs evaluate', function () {
    var r = sofuu.sandbox.run(
      'interface P { x: number } type T<A> = A[]; enum E { A, B = 4 } ' +
      'const p: P = { x: 2 }; const a: T<number> = [1]; ' +
      'function id<X>(v: X): X { return v } ' +
      'id<number>(p.x) + a.length + E.B'
    );
    assert(r.success, r.error || 'TS failed');
    assert(r.result === 7, 'expected 7, got ' + r.result);
    return 'enum+iface+generic => ' + r.result;
  });

  await check('E2 ts:false keeps annotations as a syntax error', function () {
    var r = sofuu.sandbox.run('const x: number = 1; x', { ts: false });
    assert(!r.success && r.errorType === 'syntax', 'got ' + r.errorType);
    return 'syntax';
  });

  await check('E3 JSX transpiles to a vdom tree', function () {
    var r = sofuu.sandbox.run(
      'const el = <div id="d"><b>hi</b>{2 + 2}</div>; [el.type, el.props.id, el.children[1]];',
      { jsx: true }
    );
    assert(r.success, r.error || 'JSX failed');
    assert(JSON.stringify(r.result) === '["div","d",4]', JSON.stringify(r.result));
    return JSON.stringify(r.result);
  });

  /* ── 6. Callable bridges ───────────────────────────────────── */

  await check('F1 sync bridge JSON round-trip', function () {
    var sb = sofuu.sandbox.create({
      globals: {
        echo: function (x) { return { got: x, n: [1, 2, 3], nil: null }; },
        sum: function () { var s = 0; for (var i = 0; i < arguments.length; i++) s += arguments[i]; return s; },
      },
    });
    var a = sb.run('echo({k: "v"}).got.k');
    assert(a.success && a.result === 'v', JSON.stringify(a));
    var b = sb.run('sum(1, 2, 3, 4)');
    assert(b.result === 10, 'variadic: ' + b.result);
    var c = sb.run('echo([1,2]).n[2] + "," + (echo(1).nil === null)');
    assert(c.result === '3,true', 'nested: ' + c.result);
    sb.close();
    return 'ok';
  });

  await check('F2 async bridge + Promise.all', async function () {
    var sb = sofuu.sandbox.create({
      globals: { twice: async function (n) { return n * 2; } },
    });
    var r = await sb.run('const [a, b] = await Promise.all([twice(3), twice(20)]); return a + b;');
    assert(r.success && r.result === 46, JSON.stringify(r));
    sb.close();
    return '46';
  });

  await check('F3 bridge over real host I/O parks and resumes', async function () {
    var path = '/tmp/sofuu_sandbox_stress_io.txt';
    await sofuu.fs.writeFile(path, 'io-ok\n');
    var sb = sofuu.sandbox.create({
      globals: {
        read: async function (p) { return (await sofuu.fs.readFile(p, 'utf8')).trim(); },
      },
    });
    var r = await sb.run("const t = await read('" + path + "'); return t + '!';");
    assert(r.success, JSON.stringify(r));
    assert(r.result === 'io-ok!', 'got ' + r.result);
    var after = sb.run('1 + 1');
    assert(after.success && after.result === 2, 'session must be usable after a parked run');
    sb.close();
    return 'parked+resumed';
  });

  await check('F4 bridge throw is typed and non-fatal', function () {
    var sb = sofuu.sandbox.create({ globals: { boom: function () { throw new Error('host-said-no'); } } });
    var r = sb.run('boom()');
    assert(!r.success && /host-said-no/.test(r.error || ''), JSON.stringify(r));
    assert(sb.run('2 + 2').result === 4, 'realm survived');
    sb.close();
    return 'typed';
  });

  await check('F5 non-serializable bridge result is a typed error', function () {
    var sb = sofuu.sandbox.create({
      globals: { cyc: function () { var o = {}; o.self = o; return o; } },
    });
    var r = sb.run('cyc()');
    assert(!r.success && /serializ/i.test(r.error || ''), JSON.stringify(r));
    sb.close();
    return 'E_BRIDGE-ish';
  });

  await check('F6 a parked session refuses a concurrent run (E_BUSY)', async function () {
    var path = '/tmp/sofuu_sandbox_stress_io.txt';
    var sb = sofuu.sandbox.create({
      globals: { read: async function (p) { return await sofuu.fs.readFile(p, 'utf8'); } },
    });
    var pending = sb.run("await read('" + path + "')");
    var busy = null;
    try { sb.run('1 + 1'); } catch (e) { busy = e; }
    assert(busy && /E_BUSY/.test(busy.message || ''), 'expected E_BUSY');
    await pending;
    assert(sb.run('1 + 1').result === 2, 'usable after resume');
    sb.close();
    return 'E_BUSY then resumed';
  });

  await check('F7 close() during a parked run settles without crashing', async function () {
    var sb = sofuu.sandbox.create({ globals: { later: async function () { return 1; } } });
    var p = sb.run('await later()');
    sb.close();
    var r = await p;
    assert(r && r.success, 'parked run must still settle');
    return 'deferred teardown';
  });

  /* ── 7. Console routing ────────────────────────────────────── */

  await check('G1 console modes and caps', function () {
    var cap = sofuu.sandbox.run('console.log("a",1); console.warn("b"); 1');
    assert(cap.logs.length === 2 && cap.logs[0] === 'a 1', JSON.stringify(cap.logs));
    var ign = sofuu.sandbox.run('console.log("x"); 1', { console: 'ignore' });
    assert(ign.logs.length === 0, 'ignore captured ' + ign.logs.length);
    var seen = [];
    var cb = sofuu.sandbox.run('console.error("e", 9); 1', { console: function (lvl) { seen.push(lvl); } });
    assert(cb.success && seen.length === 1 && seen[0] === 'error', JSON.stringify(seen));
    var capped = sofuu.sandbox.run('for (let i=0;i<100;i++) console.log(i); 1', { maxLogs: 3 });
    assert(capped.logs.length === 3, 'cap: ' + capped.logs.length);
    return 'capture/ignore/callback/cap';
  });

  /* ── 8. Presets ───────────────────────────────────────────── */

  await check('H1 presets behave as declared', function () {
    var strict = sofuu.sandbox.run('Object.prototype.q = 1; ({}).q === undefined', { preset: 'strict' });
    assert(strict.result === true, 'strict must freeze');
    var compute = sofuu.sandbox.run('let s=0; for(let i=0;i<6000000;i++) s++; s', { preset: 'compute' });
    assert(compute.success && compute.result === 6000000, 'compute must be unmetered');
    var worker = sofuu.sandbox.run('9 * 9', { preset: 'worker' });
    assert(worker.result === 81, 'worker preset');
    var over = sofuu.sandbox.run('while(true){}', { preset: 'worker', timeoutMs: 20 });
    assert(over.errorType === 'timeout', 'explicit option must beat the preset');
    return 'strict/compute/worker';
  });

  /* ── 9. Soak ──────────────────────────────────────────────── */

  await check('I1 2000 mixed runs stay healthy', async function () {
    var sb = sofuu.sandbox.create({ globals: { id: async function (x) { return x; } } });
    var okRuns = 0;
    var rssBefore = (typeof process !== 'undefined' && process.memoryUsage)
      ? process.memoryUsage().rss || process.memoryUsage().heapUsed : 0;
    for (var i = 0; i < 1000; i++) {
      var a = await sb.run('return await id(' + i + ')');
      var b = sb.run('(' + i + ' * 2)');
      if (a.success && a.result === i && b.success && b.result === i * 2) okRuns++;
    }
    for (var j = 0; j < 1000; j++) {
      var c = sofuu.sandbox.run('(' + j + ' + 1)', { globals: { id: function (x) { return x; } } });
      if (c.success && c.result === j + 1) okRuns++;
    }
    var rssAfter = (typeof process !== 'undefined' && process.memoryUsage)
      ? process.memoryUsage().rss || process.memoryUsage().heapUsed : 0;
    assert(okRuns === 2000, 'healthy runs: ' + okRuns + '/2000');
    var growthMB = rssBefore ? Math.round((rssAfter - rssBefore) / (1024 * 1024)) : 0;
    if (rssBefore) {
      assert(growthMB < 256, 'RSS grew ' + growthMB + 'MB over 2000 runs');
      return '2000/2000, rss+' + growthMB + 'MB';
    }
    return '2000/2000 (rss unavailable)';
  });

  /* ── Summary ──────────────────────────────────────────────── */
  console.log('\n========================================================');
  console.log((fail === 0 ? '🎉 ALL SANDBOX STRESS CASES PASSED' : '❌ SANDBOX STRESS FAILURES') +
    ' — ' + pass + ' passed, ' + fail + ' failed');
  console.log('========================================================');
  if (fail > 0) process.exit(1);
})().catch(function (e) {
  console.error('Stress runner crashed:', (e && e.stack) || e);
  process.exit(1);
});
