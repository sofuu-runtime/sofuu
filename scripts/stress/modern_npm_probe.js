// scripts/stress/modern_npm_probe.js — PLAN-MODERN-NPM acceptance probe.
//
// Runs from a fixture directory prepared by modern_npm_e2e.sh, which creates:
//
//   node_modules/modern-pkg/     exports: import/default, ./feature,
//                                ./locales/* wildcard, no require condition
//   node_modules/dual-pkg/       exports: require vs import (separate builds)
//   node_modules/native-thing/   main points at build/x.node (a fake addon)
//
// and exports SOFUU_SSE_URL pointing at a chunked text/event-stream server.
//
// Gates (exit 1 on any failure):
//   A. conditional exports: subpath, wildcard, conditions, CJS `require`
//      condition, deep-import fallback
//   B. web streams: response.body is a ReadableStream, getReader() yields
//      Uint8Array chunks INCREMENTALLY, pipeThrough(TextDecoderStream) and
//      for-await work, `tee()` fans out, abortless cancel is safe
//   C. diagnostics: .node addon, absent Node built-ins, require(esm),
//      missing package — each a coded error with advice, never a Rust panic
//
// Usage: SOFUU_SSE_URL=http://127.0.0.1:PORT/stream ./sofuu run this-file.js

'use strict';

var pass = 0, fail = 0;
function ok(label, cond, extra) {
  if (cond) { console.log('\x1b[32m  ✓\x1b[0m ' + label); pass++; }
  else { console.log('\x1b[31m  ✗\x1b[0m ' + label + (extra ? ' — ' + extra : '')); fail++; }
}
function errOf(fn) { try { fn(); return null; } catch (e) { return e; } }
async function errOfAsync(fn) { try { await fn(); return null; } catch (e) { return e; } }
function firstLine(e) { return String((e && e.message) || e).split('\n')[0]; }

(async function () {
  console.log('\n=== A. conditional exports ===');

  var root;
  try {
    root = (await import('modern-pkg')).default;
    ok('exports "." with an import condition', root && root.where === 'index.mjs', JSON.stringify(root));
  } catch (e) { ok('exports "." with an import condition', false, firstLine(e)); }

  try {
    var feature = await import('modern-pkg/feature');
    ok('exports exact subpath "./feature"', feature.feature === 'feature-mjs');
  } catch (e) { ok('exports exact subpath "./feature"', false, firstLine(e)); }

  try {
    var fr = await import('modern-pkg/locales/fr');
    ok('exports wildcard "./locales/*"', fr.fr === 'bonjour');
  } catch (e) { ok('exports wildcard "./locales/*"', false, firstLine(e)); }

  var deep = errOf(function () { return require('modern-pkg/extra.js'); });
  ok('deep import without an exports entry still resolves', deep === null && require('modern-pkg/extra.js') === 'deep-import',
     deep && firstLine(deep));

  var dualCjs = errOf(function () { return require('dual-pkg'); });
  ok('require() picks the "require" condition build',
     dualCjs === null && require('dual-pkg').flavor === 'cjs-build',
     dualCjs && firstLine(dualCjs));

  var dualEsm = await errOfAsync(function () { return import('dual-pkg'); });
  ok('import picks the "import" condition build',
     dualEsm === null && (await import('dual-pkg')).flavor === 'esm-build',
     dualEsm && firstLine(dualEsm));

  console.log('\n=== B. WHATWG web streams ===');

  ok('ReadableStream / WritableStream / TransformStream are installed',
     typeof ReadableStream === 'function' && typeof WritableStream === 'function' &&
     typeof TransformStream === 'function' && typeof TextDecoderStream === 'function');

  var manual = new ReadableStream({
    start: function (c) { c.enqueue('a'); c.enqueue('b'); c.close(); }
  });
  var reader = manual.getReader();
  ok('getReader() locks the stream', manual.locked === true);
  var r1 = await reader.read();
  var r2 = await reader.read();
  var r3 = await reader.read();
  ok('reader.read() yields queued chunks then done',
     r1.value === 'a' && r2.value === 'b' && r3.done === true);

  var teed = new ReadableStream({ start: function (c) { c.enqueue(1); c.enqueue(2); c.close(); } }).tee();
  var a = [], b = [];
  for await (var x of teed[0]) a.push(x);
  for await (var y of teed[1]) b.push(y);
  ok('tee() fans out to both branches', a.join(',') === '1,2' && b.join(',') === '1,2');

  // Multi-byte UTF-8 split across chunk boundaries.
  var split = new ReadableStream({
    start: function (c) {
      c.enqueue(new Uint8Array([0x68, 0xc3]));   // "h" + first byte of é
      c.enqueue(new Uint8Array([0xa9, 0x6c]));   // second byte + "l"
      c.close();
    }
  });
  var text = '';
  for await (var s of split.pipeThrough(new TextDecoderStream())) text += s;
  ok('TextDecoderStream keeps split multi-byte characters intact', text === 'hél', JSON.stringify(text));

  var url = (typeof process !== 'undefined' && process.env && process.env.SOFUU_SSE_URL) || '';
  if (!url) {
    ok('SOFUU_SSE_URL is set (chunked server)', false, 'run through modern_npm_e2e.sh');
  } else {
    var res = await fetch(url);
    ok('fetch() body is a ReadableStream', res.body && res.body.constructor && res.body.constructor.name === 'ReadableStream',
       res.body && res.body.constructor && res.body.constructor.name);
    ok('response.body is stable across accesses', res.body === res.body);

    var t0 = Date.now();
    var chunks = 0;
    var allBytes = 0;
    var rd = res.body.getReader();
    for (;;) {
      var step = await rd.read();
      if (step.done) break;
      chunks++;
      allBytes += step.value.length;
      if (!(step.value instanceof Uint8Array)) { ok('chunks are Uint8Array', false, typeof step.value); }
    }
    var elapsed = Date.now() - t0;
    ok('getReader() streams 5 discrete chunks', chunks === 5, 'chunks=' + chunks);
    ok('chunks arrive incrementally, not in one buffered blob', elapsed >= 100, 'elapsed=' + elapsed + 'ms');
    ok('byte count is the sum of the chunks', allBytes > 0);

    var res2 = await fetch(url);
    var acc = '';
    for await (var txt of res2.body.pipeThrough(new TextDecoderStream())) acc += txt;
    ok('pipeThrough(TextDecoderStream) + for-await sees every event',
       (acc.match(/data: /g) || []).length === 5, JSON.stringify(acc.slice(0, 40)));
  }

  console.log('\n=== C. smart diagnostics ===');

  var direct = errOf(function () { return require('native-thing/build/x.node'); });
  ok('.node file is refused with ERR_SOFUU_NATIVE_ADDON',
     direct && direct.code === 'ERR_SOFUU_NATIVE_ADDON', direct && (direct.code || firstLine(direct)));
  ok('.node message names the zero-JIT reason and the package', direct && /zero-JIT/.test(direct.message));

  var asMain = errOf(function () { require('native-thing'); });
  ok('a package whose main is a .node is refused the same way',
     asMain && asMain.code === 'ERR_SOFUU_NATIVE_ADDON', asMain && (asMain.code || firstLine(asMain)));

  for (var i = 0; i < 3; i++) {
    var name = ['child_process', 'worker_threads', 'v8'][i];
    var e1 = errOf(function () { return require(name); });
    ok('require(' + name + ') is ERR_SOFUU_UNSUPPORTED_NODE_API',
       e1 && e1.code === 'ERR_SOFUU_UNSUPPORTED_NODE_API', e1 && (e1.code || firstLine(e1)));
  }
  var esmApi = await errOfAsync(function () { return import('v8'); });
  ok('import "v8" carries the same code through the ESM loader',
     esmApi && esmApi.code === 'ERR_SOFUU_UNSUPPORTED_NODE_API', esmApi && (esmApi.code || firstLine(esmApi)));

  var esmRequire = errOf(function () { return require('dual-pkg/esm/index.mjs'); });
  ok('require() of an ES module is ERR_SOFUU_REQUIRE_ESM',
     esmRequire && esmRequire.code === 'ERR_SOFUU_REQUIRE_ESM', esmRequire && (esmRequire.code || firstLine(esmRequire)));

  var missing = errOf(function () { return require('not-installed-pkg'); });
  ok('a missing package explains how to install it',
     missing && missing.code === 'ERR_SOFUU_RESOLUTION' && /sofuu add not-installed-pkg/.test(missing.message),
     missing && firstLine(missing));

  var subpath = errOf(function () { return require('modern-pkg/nope'); });
  ok('a missing subpath lists the published subpaths',
     subpath && subpath.code === 'ERR_SOFUU_RESOLUTION' && /published subpaths/.test(subpath.message),
     subpath && firstLine(subpath));

  var nativePkg = errOf(function () { return require('sharp'); });
  ok('a known native package gets an alternative recommendation',
     nativePkg && /Recommendation:/.test(nativePkg.message), nativePkg && firstLine(nativePkg));

  console.log('\n' + (fail === 0 ? '\x1b[32m' : '\x1b[31m') +
              'modern-npm: ' + pass + ' passed, ' + fail + ' failed\x1b[0m\n');
  if (fail !== 0) process.exit(1);
})();
