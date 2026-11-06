/* scripts/bench/event_loop_latency.js — Stabilization Freeze Pass 3.1
 *
 * Measures the primary QuickJS+libuv event loop while `sofuu.serve`'s REAL
 * HTTP handler drives many concurrent streaming turns. The inference seam is
 * the only fake: each chunk resolves on a macrotask, like a real provider
 * stream, so the test measures the handler's own blocking behaviour rather
 * than a model.
 *
 * Gate: the loop must never go >16 ms without servicing timers (the 60 FPS /
 * hard-realtime heartbeat ceiling in RULES.md §3.6).
 *
 *   ./sofuu run scripts/bench/event_loop_latency.js
 *
 * Knobs (environment):
 *   SOFUU_LOOP_CONCURRENCY  concurrent turns (default 32)
 *   SOFUU_LOOP_CHUNKS       chunks per turn (default 128)
 *   SOFUU_LOOP_CHUNK_MS     per-chunk delay in ms (default 1)
 *   SOFUU_LOOP_BUDGET_MS    pass threshold (default 16)
 */
(function () {
  'use strict';

  var CONCURRENCY = parseInt(typeof process !== 'undefined' && process.env.SOFUU_LOOP_CONCURRENCY || '32', 10);
  var CHUNKS = parseInt(typeof process !== 'undefined' && process.env.SOFUU_LOOP_CHUNKS || '128', 10);
  var CHUNK_MS = parseInt(typeof process !== 'undefined' && process.env.SOFUU_LOOP_CHUNK_MS || '1', 10);
  var BUDGET_MS = parseInt(typeof process !== 'undefined' && process.env.SOFUU_LOOP_BUDGET_MS || '16', 10);
  var SAMPLE_MS = 5;

  function mkRes() {
    var out = { status: 0, headers: {}, chunks: [], ended: false };
    return {
      writeHead: function (s, h) { out.status = s; out.headers = h || {}; },
      write: function (s) { out.chunks.push(String(s)); },
      end: function (s) { if (s !== undefined) out.chunks.push(String(s)); out.ended = true; },
      _out: out,
    };
  }

  function mkReq(body) {
    return {
      method: 'POST',
      url: '/v1/chat/completions',
      body: body,
      headers: { authorization: 'Bearer tok' },
    };
  }

  /* The only substitution: an async stream whose chunks arrive on macrotasks,
   * exactly as a provider stream's I/O wakeups do. */
  sofuu.ai.stream = function () {
    var i = 0;
    return {
      next: function () {
        if (i >= CHUNKS) return Promise.resolve({ value: undefined, done: true });
        i++;
        var text = 'chunk-' + i + ':' + new Array(48).join('x');
        return new Promise(function (resolve) {
          setTimeout(function () {
            resolve({ value: { text: text, think: '' }, done: false });
          }, CHUNK_MS);
        });
      },
      get usage() { return { promptTokens: 12, completionTokens: CHUNKS }; },
      get toolCalls() { return []; },
      get aborted() { return false; },
    };
  };

  var handler = sofuu.serve.createHandler({
    modelPath: '/tmp/sofuu-loop.gguf',
    models: [{ id: 'loop.gguf', path: '/tmp/sofuu-loop.gguf' }],
    token: 'tok',
    host: '127.0.0.1',
    port: 0,
    version: 'bench',
    brainPath: '',
  });

  var last = Date.now();
  var maxLag = 0;
  var samples = [];
  var timer = setInterval(function () {
    var now = Date.now();
    var lag = now - last - SAMPLE_MS;
    if (lag < 0) lag = 0;
    if (lag > maxLag) maxLag = lag;
    samples.push(lag);
    last = now;
  }, SAMPLE_MS);

  var body = JSON.stringify({
    model: 'loop.gguf',
    messages: [{ role: 'user', content: 'measure the loop' }],
    stream: true,
  });

  var started = Date.now();
  var jobs = [];
  for (var i = 0; i < CONCURRENCY; i++) {
    jobs.push(handler(mkReq(body), mkRes()));
  }

  Promise.all(jobs).then(function () {
    clearInterval(timer);
    var elapsed = Date.now() - started;
    samples.sort(function (a, b) { return a - b; });
    var p95 = samples.length ? samples[Math.floor(samples.length * 0.95)] : 0;
    var mean = 0;
    for (var s = 0; s < samples.length; s++) mean += samples[s];
    mean = samples.length ? Math.round((mean / samples.length) * 10) / 10 : 0;
    console.log('event-loop latency: ' + CONCURRENCY + ' concurrent turns × ' +
      CHUNKS + ' chunks, ' + elapsed + ' ms wall');
    console.log('  samples=' + samples.length + ' mean=' + mean + ' ms p95=' + p95 +
      ' ms max=' + maxLag + ' ms (budget ' + BUDGET_MS + ' ms)');
    if (maxLag > BUDGET_MS) {
      console.log('FAIL: the event loop was blocked for ' + maxLag + ' ms');
      process.exit(1);
    }
    console.log('PASS: no turn blocked the loop beyond the ' + BUDGET_MS + ' ms ceiling');
    process.exit(0);
  }, function (e) {
    clearInterval(timer);
    console.log('FAIL: handler rejected: ' + String((e && e.message) || e));
    process.exit(1);
  });
})();
