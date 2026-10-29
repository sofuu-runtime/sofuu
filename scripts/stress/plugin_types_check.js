// scripts/stress/plugin_types_check.js — the typings/runtime parity gate.
//
// Two artefacts describe the same contract: the runtime (src/js/plugins.js,
// shipped as bytecode) and the SDK typings (sofuu.d.ts, shipped to npm).
// They drift the moment someone adds a runtime function and forgets the
// declaration — a type-checked plugin then fails at runtime with "not a
// function". This gate compares both directions and also proves that every
// cookbook plugin in plugins/*.ts strips to PARSEABLE JavaScript, which is
// the only "compile" step a .ts plugin gets in this runtime.
//
//   make plugin-check
//   ./sofuu run scripts/stress/plugin_types_check.js
(async function () {
  'use strict';

  var P = (typeof sofuu !== 'undefined' && sofuu.plugins) || null;
  var pass = 0, fail = 0;
  function ok(name, detail) { pass++; console.log('  PASS ' + name + (detail ? '  [' + detail + ']' : '')); }
  function bad(name, detail) { fail++; console.log('  FAIL ' + name + '  [' + detail + ']'); }
  async function check(name, fn) {
    try { var d = await fn(); ok(name, d); }
    catch (e) { bad(name, (e && e.message) || e); }
  }

  /* The plugin surface PLUGIN_PLAN.md promises. Every entry must exist at
   * runtime AND be declared in the typings. */
  var RUNTIME_FUNCS = [
    'defineTool', 'defineProvider', 'defineStorage', 'defineVectorStore', 'defineService',
    'useOpenAPI', 'discover', 'discoverAll', 'loadManifest', 'list', 'invoke',
    'registerProvider', 'getProvider', 'listProviders', 'resolveProvider',
    'completeVia', 'streamVia', 'attachVectorStore', 'attachStorage', 'mirrorStats',
    'flushMirror', 'emitTurnStart', 'emitTurnEnd', 'emitToolCall', 'emitToolResult',
    'startServices', 'stopServices', 'ingestMcp', 'capabilities',
    'cliList', 'cliValidate', 'cliRun', 'cliTest', 'cliInit',
    'resetCircuit', 'circuitState'
  ];
  var RUNTIME_OBJECTS = ['z', 'TOOLS', 'PROVIDERS', 'STORES', 'SERVICES', 'CIRCUITS', 'modules', '_stripTs', '_yaml'];

  /* Typings names that must exist for a plugin author to compile. */
  var DTS_REQUIRED = [
    'defineTool', 'defineProvider', 'defineStorage', 'defineVectorStore', 'defineService',
    'useOpenAPI', 'z', 'SofuuRuntime', 'ToolSpec', 'ToolContext', 'ProviderSpec',
    'ProviderRequest', 'ProviderResult', 'StorageSpec', 'VectorStoreSpec', 'ServiceSpec',
    'TurnEvent', 'ToolCallEvent', 'OpenAPIOptions', 'SofuuErrorCode', 'SofuuError', 'ZodTypeAny'
  ];

  /* sofuu.fs.readFile is a PROMISE in the real runtime (the host pumps the
   * event loop), so every read here is awaited — asking for a sync read made
   * all four file-based gates fail with 'sync read expected' against a live
   * binary while passing under a stub. */
  async function readText(path) {
    if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') throw new Error('sofuu.fs unavailable');
    var t = sofuu.fs.readFile(path);
    if (t && typeof t.then === 'function') t = await t;
    return String(t);
  }

  console.log('=== Plugin typings/runtime parity ===\n');

  await check('runtime surface is complete', function () {
    if (!P) throw new Error('sofuu.plugins missing');
    var missing = [];
    for (var i = 0; i < RUNTIME_FUNCS.length; i++) {
      if (typeof P[RUNTIME_FUNCS[i]] !== 'function') missing.push(RUNTIME_FUNCS[i]);
    }
    for (var j = 0; j < RUNTIME_OBJECTS.length; j++) {
      if (P[RUNTIME_OBJECTS[j]] === undefined || P[RUNTIME_OBJECTS[j]] === null) missing.push(RUNTIME_OBJECTS[j]);
    }
    if (missing.length) throw new Error('missing: ' + missing.join(', '));
    return RUNTIME_FUNCS.length + ' functions + ' + RUNTIME_OBJECTS.length + ' objects';
  });

  await check('sofuu.d.ts declares the documented surface', async function () {
    var dts = await readText('sofuu.d.ts');
    var missing = [];
    for (var i = 0; i < DTS_REQUIRED.length; i++) {
      var nm = DTS_REQUIRED[i];
      var re = new RegExp('(^|\\n)\\s*export\\s+(declare\\s+)?(function|const|interface|type|class|namespace)\\s+' + nm + '\\b');
      var re2 = new RegExp('(^|\\n)\\s*(interface|type|declare const)\\s+' + nm + '\\b');
      if (!re.test(dts) && !re2.test(dts)) missing.push(nm);
    }
    if (missing.length) throw new Error('not declared: ' + missing.join(', '));
    return DTS_REQUIRED.length + ' declarations';
  });

  await check('sofuu.d.ts is structurally sound', async function () {
    var dts = await readText('sofuu.d.ts');
    var opens = (dts.match(/\{/g) || []).length;
    var closes = (dts.match(/\}/g) || []).length;
    if (opens !== closes) throw new Error('brace imbalance: ' + opens + ' vs ' + closes);
    var seen = {}, dupes = [];
    (dts.match(/export\s+(?:declare\s+)?(?:interface|type|class|const|function)\s+([A-Za-z_$][\w$]*)/g) || [])
      .forEach(function (m) {
        var nm = m.replace(/^export\s+(declare\s+)?(interface|type|class|const|function)\s+/, '');
        if (seen[nm]) dupes.push(nm);
        seen[nm] = 1;
      });
    if (dupes.length) throw new Error('duplicate exports: ' + dupes.join(', '));
    return opens + ' blocks, ' + Object.keys(seen).length + ' exports';
  });

  await check('every runtime function is named in the typings', async function () {
    var dts = await readText('sofuu.d.ts');
    var missing = [];
    for (var i = 0; i < RUNTIME_FUNCS.length; i++) {
      var nm = RUNTIME_FUNCS[i];
      /* A callable may be declared as a function, or as a property of the
       * PluginHost/runtime interface (defineTool: typeof defineTool). */
      if (dts.indexOf(nm) < 0) missing.push(nm);
    }
    if (missing.length) throw new Error('undocumented: ' + missing.join(', '));
    return RUNTIME_FUNCS.length + ' checked';
  });

  await check('cookbook plugins strip to parseable JavaScript', async function () {
    var dir = 'plugins';
    var entries = [];
    var rd = sofuu.fs.readdir || sofuu.fs.listDir;
    try { entries = rd ? (rd.call(sofuu.fs, dir) || []) : []; if (entries && typeof entries.then === 'function') entries = await entries; } catch (e) { entries = []; }
    entries = (entries || []).map(function (e) { return String((e && e.name) || e); });
    if (!entries.length) throw new Error('no plugins/ directory contents');
    var ts = entries.filter(function (e) { return /\.ts$/.test(e); });
    if (!ts.length) throw new Error('no .ts cookbook plugins found in plugins/');
    if (typeof P._stripTs !== 'function') throw new Error('P._stripTs missing');
    var bad = [], done = [];
    for (var i = 0; i < ts.length; i++) {
      var src = await readText(dir + '/' + ts[i]);
      var exec = src.replace(/import(?:\s+[\s\S]*?\s+from)?\s+['"][^'"]+['"]\s*;?/g, '')
                    .replace(/export\s+default\s+/g, 'var __x = ')
                    .replace(/export\s+(const|let|var|function|class)\s+/g, '$1 ');
      try {
        var stripped = P._stripTs(exec);
        new Function('return (function(){' + stripped + '})')();
        done.push(ts[i]);
      } catch (e2) {
        bad.push(ts[i] + ': ' + ((e2 && e2.message) || e2));
      }
    }
    if (bad.length) throw new Error(bad.join(' | '));
    return done.length + ' plugins: ' + done.join(', ');
  });

  console.log('\n=== SUMMARY ===');
  console.log('parity checks PASS=' + pass + ' FAIL=' + fail);
  console.log('PLUGIN_TYPES_RECEIPT ' + JSON.stringify({ passed: pass, failed: fail }));
  try { if (typeof process !== 'undefined' && process.exit) process.exit(fail > 0 ? 1 : 0); } catch (eX) {}
})();
