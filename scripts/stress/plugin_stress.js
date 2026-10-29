// scripts/stress/plugin_stress.js — adversarial stress suite for the
// Universal Plugin & Extensibility Architecture (PLUGIN_PLAN.md).
//
// This is not a unit test: every case drives the plugin runtime the way an
// untrusted caller would and records a receipt (counts, latencies, outcome
// codes). It exits non-zero on any FAIL, so it doubles as a gate.
//
//   ./sofuu run scripts/stress/plugin_stress.js
//   ./sofuu run scripts/stress/plugin_stress.js --fast   (fewer iterations)
(function () {
  'use strict';

  var P = (typeof sofuu !== 'undefined' && sofuu.plugins) || null;
  if (!P) {
    console.error('FATAL: sofuu.plugins is not loaded in this build');
    if (typeof process !== 'undefined') process.exit(2);
    return;
  }

  var FAST = false;
  try {
    if (typeof process !== 'undefined' && process.argv && String(process.argv.join(' ')).indexOf('--fast') >= 0) FAST = true;
  } catch (eA) {}
  function N(n) { return FAST ? Math.max(1, Math.ceil(n / 10)) : n; }

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
  function codeOf(res) { return (res && res.error && res.error.code) || (res && res.code) || ''; }
  function eq(a, b, msg) { if (a !== b) throw new Error((msg || 'mismatch') + ': expected ' + JSON.stringify(b) + ' got ' + JSON.stringify(a)); }

  function percentile(arr, p) {
    if (!arr.length) return 0;
    var a = arr.slice().sort(function (x, y) { return x - y; });
    var i = Math.min(a.length - 1, Math.max(0, Math.ceil((p / 100) * a.length) - 1));
    return a[i];
  }

  var PORT = 13371;
  var server = null;
  function startServer() {
    return new Promise(function (resolve) {
      try {
        server = sofuu.http.createServer(function (req, res) {
          var url = String(req.url || '');
          if (url.indexOf('/pets/') === 0) {
            var id = url.split('/')[2] || '0';
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ id: id, name: 'pet' + id, tag: 'dog' }));
            return;
          }
          if (url.indexOf('/pets') === 0) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify([{ id: '1', name: 'rex' }]));
            return;
          }
          res.writeHead(200, { 'Content-Type': 'text/plain' });
          res.end('ok');
        });
        server.listen(PORT, '127.0.0.1');
        setTimeout(function () { resolve(); }, 150);
      } catch (e) { resolve(); }
    });
  }
  function stopServer() {
    try { if (server && server.stop) server.stop(); } catch (e1) {}
    try { if (server && server.close) server.close(); } catch (e2) {}
  }

  async function shields() {
    console.log('\n=== Cat 1: defineTool / 5 shields ===');
    var Z = P.z;

    await check('S1.1 schema rejects malformed args under concurrency', async function () {
      var t = P.defineTool({
        name: 'stress_echo', description: 'echo',
        schema: Z.object({ who: Z.string().describe('name'), n: Z.number().optional() }),
        async run(args) { return { hello: args.who, n: args.n || 0 }; }
      });
      var bad = [], good = [];
      for (var i = 0; i < N(1000); i++) bad.push(t.execute({ who: 123 }));
      for (var j = 0; j < N(1000); j++) good.push(t.execute({ who: 'w' + j }));
      var br = await Promise.all(bad);
      var gr = await Promise.all(good);
      var badOk = br.every(function (r) { return r && r.ok === false && codeOf(r) === 'E_SCHEMA'; });
      var goodOk = gr.every(function (r) { return r && r.ok === true; });
      assert(badOk, 'not every malformed call returned E_SCHEMA');
      assert(goodOk, 'a well-formed call failed');
      return 'bad=' + br.length + ' good=' + gr.length;
    });

    await check('S1.2 output schema is enforced', async function () {
      var t = P.defineTool({
        name: 'stress_out', description: 'declares an output contract',
        schema: Z.object({}), output: Z.object({ total: Z.number() }),
        async run() { return { total: 'not-a-number' }; }
      });
      var r = await t.execute({});
      assert(r.ok === false, 'an output-schema violation must fail the call');
      eq(codeOf(r), 'E_SCHEMA', 'output violation code');
      return codeOf(r);
    });

    await check('S1.3 a garbage schema is rejected at define time', async function () {
      var threw = null;
      try { P.defineTool({ name: 'stress_badschema', schema: 'nonsense', async run() { return 1; } }); }
      catch (e) { threw = e; }
      assert(threw, 'defineTool accepted a non-schema');
      eq(threw.code, 'E_SCHEMA', 'define-time code');
      return threw.code;
    });

    await check('S2.1 SSRF matrix: every hostile target is denied', async function () {
      var targets = [
        'http://127.0.0.1:' + PORT + '/x', 'http://localhost:' + PORT + '/x',
        'http://0.0.0.0/x', 'http://169.254.169.254/latest/meta-data/',
        'http://10.0.0.1/x', 'http://192.168.1.1/x', 'http://172.16.0.1/x',
        'http://[::1]/x', 'http://metadata.google.internal/x',
        'http://2130706433/x', 'http://0x7f000001/x', 'http://foo.localhost/x',
        'file:///etc/passwd', 'data:text/html,hi', 'ftp://example.com/x',
        'http://user:pass@example.com/x'
      ];
      var t = P.defineTool({
        name: 'stress_ssrf', description: 'attempts an egress',
        schema: Z.object({ url: Z.string() }),
        async run(args, ctx) { await ctx.fetch(args.url); return 'reached'; }
      });
      var denied = 0, leaked = [];
      for (var i = 0; i < targets.length; i++) {
        var r = await t.execute({ url: targets[i] });
        if (r.ok === false && codeOf(r) === 'E_DENIED') denied++;
        else leaked.push(targets[i] + ' -> ' + (codeOf(r) || 'ok'));
      }
      assert(leaked.length === 0, 'leaked: ' + leaked.join(', '));
      eq(denied, targets.length, 'denied count');
      return denied + '/' + targets.length + ' denied';
    });

    await check('S2.2 an explicit net allowlist admits the loopback server', async function () {
      var t = P.defineTool({
        name: 'stress_allow', description: 'allowlisted egress',
        schema: Z.object({ url: Z.string() }),
        permissions: { net: ['127.0.0.1'] },
        async run(args, ctx) { var res = await ctx.fetch(args.url); return { status: res.status, body: await res.text() }; }
      });
      var r = await t.execute({ url: 'http://127.0.0.1:' + PORT + '/ok' });
      assert(r.ok === true, 'allowlisted loopback must pass: ' + (r.error && r.error.message));
      eq(r.data.status, 200, 'loopback status');
      var out = await t.execute({ url: 'http://169.254.169.254/x' });
      assert(out.ok === false && codeOf(out) === 'E_DENIED', 'allowlist must not open the world');
      return 'loopback ok, metadata denied';
    });

    await check('S3.1 circuit trips after threshold and fails fast', async function () {
      var calls = 0;
      var t = P.defineTool({
        name: 'stress_flaky', description: 'always throws',
        schema: Z.object({}),
        circuitBreaker: { failureThreshold: 5, resetTimeoutMs: 60000 },
        async run() { calls++; throw new Error('boom'); }
      });
      for (var i = 0; i < 5; i++) await t.execute({});
      var trip = await t.execute({});
      eq(codeOf(trip), 'E_CIRCUIT_OPEN', 'post-threshold code');
      eq(calls, 5, 'the open circuit must not reach the tool');
      return 'opened after ' + calls + ' calls';
    });

    await check('S3.2 HALF_OPEN admits exactly one trial under a concurrent burst', async function () {
      var calls = 0;
      var t = P.defineTool({
        name: 'stress_half', description: 'slow failure',
        schema: Z.object({}),
        circuitBreaker: { failureThreshold: 2, resetTimeoutMs: 60 },
        async run() { calls++; await new Promise(function (r) { setTimeout(r, 30); }); throw new Error('nope'); }
      });
      await t.execute({}); await t.execute({});
      await new Promise(function (r) { setTimeout(r, 120); });
      calls = 0;
      var burst = [];
      for (var i = 0; i < 10; i++) burst.push(t.execute({}));
      var res = await Promise.all(burst);
      eq(calls, 1, 'exactly one trial call may reach a HALF_OPEN tool');
      var opened = res.filter(function (r) { return codeOf(r) === 'E_CIRCUIT_OPEN'; }).length;
      assert(opened >= 8, 'the rest must fail fast, got ' + opened);
      return 'trials=1 fastfail=' + opened;
    });

    await check('S4.1 watchdog fires near the deadline (100 runs)', async function () {
      var t = P.defineTool({
        name: 'stress_hang', description: 'hangs',
        schema: Z.object({}), timeoutMs: 60,
        async run() { await new Promise(function (r) { setTimeout(r, 5000); }); return 'late'; }
      });
      var t0 = now();
      var codes = {};
      for (var i = 0; i < N(100); i++) {
        /* Shield 4 is under test here, so Shield 3 is reset every round —
         * otherwise the fifth consecutive timeout opens the breaker and the
         * remaining calls fast-fail with E_CIRCUIT_OPEN (correct behaviour,
         * wrong measurement). The fail-fast path has its own case, S3.1. */
        if (typeof P.resetCircuit === 'function') P.resetCircuit('stress_hang');
        var r = await t.execute({});
        var c = codeOf(r) || 'ok';
        codes[c] = (codes[c] || 0) + 1;
      }
      var elapsed = now() - t0;
      eq(codes['E_TIMEOUT'] || 0, N(100), 'every hung call must time out');
      assert(elapsed < N(100) * 400, 'watchdog too slow: ' + Math.round(elapsed) + 'ms');
      return 'timeouts=' + codes['E_TIMEOUT'] + ' avg=' + Math.round(elapsed / N(100)) + 'ms';
    });

    await check('S5.1 panic guard survives every JS failure shape', async function () {
      var shapes = {
        syncThrow: function () { throw new Error('sync'); },
        stringThrow: function () { throw 'plain string'; },
        nullThrow: function () { throw null; },
        rejected: async function () { throw new Error('async'); },
        circular: async function () { var o = {}; o.self = o; return o; },
        bigInt: async function () { return { v: (typeof BigInt === 'function') ? BigInt(1) : 1 }; }
      };
      var names = Object.keys(shapes);
      for (var i = 0; i < names.length; i++) (function (nm) {
        P.defineTool({
          name: 'stress_panic_' + nm, description: 'failure shape ' + nm,
          schema: Z.object({}), async run() { return shapes[nm](); }
        });
      })(names[i]);
      var out = [];
      for (var k = 0; k < names.length; k++) {
        var r = await P.TOOLS['stress_panic_' + names[k]].execute({});
        out.push(names[k] + '=' + (r.ok ? 'ok' : (codeOf(r) || 'E_TOOL')));
        assert(r && typeof r.ok === 'boolean', 'no envelope for ' + names[k]);
      }
      return out.join(' ');
    });

    await check('S6.1 400-way concurrent mixed load keeps envelopes intact', async function () {
      var t = P.defineTool({
        name: 'stress_conc', description: 'tiny async work',
        schema: Z.object({ i: Z.number() }),
        async run(args) { await new Promise(function (r) { setTimeout(r, 1); }); return { i: args.i }; }
      });
      var ps = [];
      for (var i = 0; i < N(400); i++) ps.push(t.execute({ i: i }));
      var res = await Promise.all(ps);
      var ok = 0;
      for (var k = 0; k < res.length; k++) if (res[k] && res[k].ok === true && res[k].data.i === k) ok++;
      eq(ok, N(400), 'every concurrent call returned its own result');
      return 'n=' + N(400);
    });
  }

  async function providers() {
    console.log('\n=== Cat 2: defineProvider / completeVia / streamVia ===');
    if (!P.defineProvider) { skip('providers', 'no provider runtime'); return; }

    await check('P1.1 a provider completes and reports usage', async function () {
      P.defineProvider({
        name: 'stress_prov', kind: 'custom', baseUrl: 'http://127.0.0.1:1',
        async complete(req) {
          return { text: 'echo:' + (req.messages[0] && req.messages[0].content), model: req.model || 'm',
                   usage: { promptTokens: 3, completionTokens: 5 } };
        }
      });
      var r = await P.completeVia('stress_prov', { messages: [{ role: 'user', content: 'hi' }], model: 'm' }, {});
      assert(r.ok === true, 'completeVia failed: ' + (r.error && r.error.message));
      eq(r.data.text, 'echo:hi', 'provider text');
      eq(r.data.usage.completionTokens, 5, 'usage passthrough');
      return 'text=' + r.data.text;
    });

    await check('P1.2 a throwing provider is normalised to E_TOOL', async function () {
      P.defineProvider({ name: 'stress_prov_bad', kind: 'custom', async complete() { throw new Error('provider down'); } });
      var r = await P.completeVia('stress_prov_bad', { messages: [] }, {});
      assert(r.ok === false, 'expected failure envelope');
      eq(codeOf(r), 'E_TOOL', 'provider failure code');
      return codeOf(r);
    });

    await check('P1.3 provider streaming replays deltas in order', async function () {
      if (typeof P.streamVia !== 'function') throw new Error('streamVia missing');
      P.defineProvider({
        name: 'stress_prov_stream', kind: 'custom',
        async stream(req, ctx, onEvent) {
          var parts = ['a', 'b', 'c'];
          for (var i = 0; i < parts.length; i++) onEvent({ type: 'delta', text: parts[i] });
          onEvent({ type: 'done', usage: { promptTokens: 1, completionTokens: 3 } });
          return { text: parts.join('') };
        }
      });
      var seen = [];
      var r = await P.streamVia('stress_prov_stream', { messages: [] }, {}, function (ev) {
        if (ev && ev.type === 'delta') seen.push(ev.text);
      });
      assert(r.ok === true, 'streamVia failed: ' + (r.error && r.error.message));
      eq(seen.join(''), 'abc', 'delta order');
      return 'deltas=' + seen.length;
    });

    await check('P1.4 150 concurrent provider calls stay isolated', async function () {
      var ps = [];
      for (var i = 0; i < N(150); i++) ps.push(P.completeVia('stress_prov', { messages: [{ role: 'user', content: 'x' + i }] }, {}));
      var r = await Promise.all(ps);
      var ok = r.filter(function (x) { return x.ok === true; }).length;
      eq(ok, N(150), 'concurrent provider calls');
      return 'n=' + N(150);
    });

    await check('P1.5 provider registry is introspectable', async function () {
      var list = (typeof P.listProviders === 'function') ? P.listProviders() : [];
      assert(list.length >= 1, 'no providers listed');
      var got = typeof P.getProvider === 'function' ? P.getProvider('stress_prov') : null;
      assert(got && got.name === 'stress_prov', 'getProvider did not return the spec');
      return 'providers=' + list.length;
    });
  }

  async function stores() {
    console.log('\n=== Cat 3: defineVectorStore / dual-write mirror ===');
    if (typeof P.attachVectorStore !== 'function') { skip('mirror', 'attachVectorStore missing'); return; }

    await check('M1.1 local write is synchronous, remote is write-behind', async function () {
      var local = { items: [], remember: function (v, t) { this.items.push({ v: v, t: t }); return true; },
                    recall: function () { return this.items.slice(0, 1).map(function (x) { return { text: x.t }; }); },
                    count: function () { return this.items.length; }, flush: function () {} };
      var remote = { written: 0, failed: 0,
        async upsert(o) { if (o.text === 'poison') throw new Error('remote down'); remote.written++; } };
      P.defineVectorStore({ name: 'stress_store', async init() {}, async upsert(o) { return remote.upsert(o); },
                            async query() { return []; } });
      var brain = P.attachVectorStore(local, 'stress_store', { batch: 32, flushMs: 20 });
      var t0 = now();
      for (var i = 0; i < N(2000); i++) brain.remember([1, 2, 3], 'fact' + i);
      var localMs = now() - t0;
      eq(local.count(), N(2000), 'local count is immediately exact');
      brain.remember([1, 2, 3], 'poison');
      await P.flushMirror();
      var st = P.mirrorStats ? P.mirrorStats()['stress_store'] : null;
      assert(st, 'mirrorStats missing');
      assert(st.failed >= 1, 'a rejected upsert must be counted, not swallowed');
      eq(st.queued, 0, 'queue drained after flush');
      return 'local=' + local.count() + ' written=' + st.written + ' failed=' + st.failed + ' localMs=' + Math.round(localMs);
    });
  }

  async function services() {
    console.log('\n=== Cat 4: defineService hooks ===');

    await check('L1.1 onTurnStart rewrites the prompt', async function () {
      if (typeof P.emitTurnStart !== 'function') throw new Error('emitTurnStart missing');
      P.defineService({ name: 'stress_redact', async onTurnStart(turn) { turn.prompt = String(turn.prompt).replace(/SECRET-\d+/g, '[redacted]'); return turn; } });
      var ev = await P.emitTurnStart({ turnId: 't1', prompt: 'key SECRET-123 here', agent: 'a', depth: 0 });
      assert(ev && typeof ev.prompt === 'string', 'emitTurnStart must return the event');
      assert(ev.prompt.indexOf('SECRET-123') < 0, 'prompt was not rewritten: ' + ev.prompt);
      return ev.prompt;
    });

    await check('L1.2 onToolCall blocks a dangerous call and mutates args', async function () {
      if (typeof P.emitToolCall !== 'function') throw new Error('emitToolCall missing');
      P.defineService({
        name: 'stress_guard',
        async onToolCall(call) {
          if (call.tool === 'bash' && String(call.args.cmd || '').indexOf('rm -rf') >= 0) {
            var e = new Error('Blocked destructive command'); e.code = 'E_DENIED'; throw e;
          }
          if (call.args && typeof call.args.cmd === 'string') call.args.cmd = call.args.cmd.replace(/TOKEN_\w+/g, '[redacted]');
        }
      });
      var blocked = null;
      try { await P.emitToolCall({ tool: 'bash', args: { cmd: 'sudo rm -rf /' }, agent: 'a', turnId: 't' }); }
      catch (e) { blocked = e; }
      assert(blocked, 'destructive command was not blocked');
      eq(blocked.code, 'E_DENIED', 'block code');
      var args = { cmd: 'curl -H x TOKEN_abc' };
      await P.emitToolCall({ tool: 'bash', args: args, agent: 'a', turnId: 't' });
      assert(args.cmd.indexOf('TOKEN_abc') < 0, 'args mutation did not apply');
      return args.cmd;
    });

    await check('L1.3 a throwing service cannot break the others', async function () {
      P.defineService({ name: 'stress_explode', async onTurnEnd() { throw new Error('service exploded'); } });
      var order = [];
      P.defineService({ name: 'stress_order', async onTurnEnd() { order.push('after-explosion'); } });
      await P.emitTurnEnd({ turnId: 't', response: 'x', usage: {} });
      assert(order.length === 1, 'a sibling service was skipped');
      return 'isolation ok';
    });

    await check('L1.4 500 hook dispatches stay under budget', async function () {
      var t0 = now();
      for (var i = 0; i < N(500); i++) await P.emitToolResult({ tool: 'read_file', args: {}, result: {}, ok: true, durationMs: 1 });
      var ms = now() - t0;
      assert(ms < 60000, 'hook dispatch too slow: ' + Math.round(ms) + 'ms');
      return (Math.round(ms / N(500) * 1000) / 1000) + 'ms/call';
    });
  }

  async function protocols() {
    console.log('\n=== Cat 5: useOpenAPI / MCP ===');

    await check('O1.1 YAML spec with refs generates callable tools', async function () {
      if (typeof P.useOpenAPI !== 'function') throw new Error('useOpenAPI missing');
      var yaml = [
        'openapi: 3.0.3',
        'info:',
        '  title: Pet API',
        '  version: "1.0.0"',
        'servers:',
        '  - url: http://127.0.0.1:' + PORT,
        'paths:',
        '  /pets/{petId}:',
        '    get:',
        '      operationId: getPet',
        '      summary: Read one pet',
        '      parameters:',
        '        - $ref: "#/components/parameters/PetId"',
        '      responses:',
        '        "200":',
        '          description: ok',
        '  /pets:',
        '    get:',
        '      operationId: listPets',
        '      summary: List pets',
        'components:',
        '  parameters:',
        '    PetId:',
        '      name: petId',
        '      in: path',
        '      required: true',
        '      schema:',
        '        type: string'
      ].join('\n');
      var tools = await P.useOpenAPI(yaml, { namespace: 'pets', baseUrl: 'http://127.0.0.1:' + PORT });
      assert(tools && tools.length >= 2, 'expected 2 tools, got ' + (tools && tools.length));
      var names = tools.map(function (t) { return t.name; }).join(',');
      assert(names.indexOf('pets_getPet') >= 0, 'getPet not generated: ' + names);
      var call = await P.invoke('pets_getPet', { petId: '7' });
      assert(call.ok === true, 'generated tool call failed: ' + (call.error && call.error.message));
      assert(String(call.data && call.data.id) === '7', 'unexpected payload: ' + JSON.stringify(call.data));
      return 'tools=' + tools.length + ' -> ' + JSON.stringify(call.data);
    });

    await check('O1.2 a malformed spec is refused with E_SCHEMA', async function () {
      var threw = null;
      try { await P.useOpenAPI({ info: { title: 'no openapi key' } }, {}); } catch (e) { threw = e; }
      assert(threw, 'a spec without openapi/swagger was accepted');
      eq(threw.code, 'E_SCHEMA', 'spec rejection code');
      return threw.code;
    });

    await check('O1.3 MCP ingestion degrades cleanly with no server', async function () {
      if (typeof P.ingestMcp !== 'function') throw new Error('ingestMcp missing');
      var r = await P.ingestMcp('no_such_server', {});
      assert(r && typeof r === 'object', 'ingestMcp returned nothing');
      if (r.ok === false) { eq(r.error.code, 'E_CAPABILITY', 'mcp failure code'); return 'E_CAPABILITY'; }
      return 'ingested=' + ((r.tools && r.tools.length) || 0);
    });
  }

  async function stripper() {
    console.log('\n=== Phase 1: stripTs fuzz ===');
    await check('T1.1 every TS sample strips to valid JS or throws typed', async function () {
      if (typeof P._stripTs !== 'function') throw new Error('P._stripTs missing');
      var BT = String.fromCharCode(96);
      var samples = [
        'const x: number = 1;',
        'function f(a: string, b?: number): void {}',
        'const f = <T>(x: T): T => x;',
        'let y = { a: 1 } as const;',
        'const s: string = ' + BT + 'tpl interface X { a: b }' + BT + ';',
        '// interface NotAType { a: b } in a comment\nconst c = 1;',
        "import type { Foo } from './x';\nexport default 1;",
        'type U = { a: string } | null;',
        'declare const g: number;',
        'class K { constructor(private a: number) {} }',
        'const r = (n: number) => n * 2;',
        'export const h = (a: Array<string>): Promise<number> => Promise.resolve(1);',
        'function g2(o: { a: string; b: number }) { return o.a; }',
        'enum E { A, B }\nexport default E;'
      ];
      var ok = 0, typed = 0, bad = [];
      for (var i = 0; i < samples.length; i++) {
        var out = null;
        try { out = P._stripTs(samples[i]); }
        catch (e) {
          if (e && e.code === 'E_SCHEMA') { typed++; continue; }
          bad.push('sample ' + i + ' threw ' + (e && e.message)); continue;
        }
        try { new Function('return (function(){' + out + '})')(); ok++; }
        catch (e2) { bad.push('sample ' + i + ' produced invalid JS: ' + (e2 && e2.message)); }
      }
      assert(bad.length === 0, bad.join(' | '));
      return 'parsed=' + ok + ' typed-reject=' + typed;
    });

    await check('T1.2 plain JS passes through unchanged', async function () {
      var js = 'function add(a, b) { return a + b; }\nvar q = add(1, 2);\nexport default q;';
      var out = P._stripTs(js);
      assert(out.indexOf('function add(a, b)') >= 0, 'plain JS was mangled: ' + out);
      return 'no-op';
    });
  }

  function receipt() {
    var lat = R.cases.map(function (c) { return c.ms; });
    var total = Date.now() - R.startedAt;
    var mem = null;
    try { if (typeof process !== 'undefined' && process.memoryUsage) mem = process.memoryUsage(); } catch (eM) {}
    var out = {
      suite: 'plugin_stress', fast: FAST,
      passed: R.passed, failed: R.failed, skipped: R.skipped, wallMs: total,
      caseMs: { p50: Math.round(percentile(lat, 50)), p99: Math.round(percentile(lat, 99)), max: Math.round(percentile(lat, 100)) },
      rssKb: mem ? Math.round((mem.rss || mem.heapUsed || 0) / 1024) : null,
      tools: Object.keys(P.TOOLS || {}).length,
      providers: Object.keys(P.PROVIDERS || {}).length,
      services: Object.keys(P.SERVICES || {}).length,
      circuits: Object.keys(P.CIRCUITS || {}).length
    };
    console.log('\n=== SUMMARY ===');
    console.log('cases=' + (R.passed + R.failed + R.skipped) + ' PASS=' + R.passed + ' FAIL=' + R.failed + ' SKIP=' + R.skipped + ' wall=' + total + 'ms');
    console.log('PLUGIN_STRESS_RECEIPT ' + JSON.stringify(out));
    return out;
  }

  (async function main() {
    console.log('=== Sofuu Plugin Stress Suite' + (FAST ? ' (--fast)' : '') + ' ===');
    console.log('plugins API: version=' + (P.version || P.VERSION || '?') + ' modules=' + ((P.modules && P.modules.length) || 0));
    await startServer();
    try {
      await shields();
      await providers();
      await stores();
      await services();
      await protocols();
      await stripper();
    } catch (e) {
      console.error('FATAL in suite: ' + ((e && e.stack) || e));
      R.failed++;
    }
    stopServer();
    var out = receipt();
    try { if (typeof process !== 'undefined' && process.exit) process.exit(out.failed > 0 ? 1 : 0); } catch (eX) {}
  })();
})();
