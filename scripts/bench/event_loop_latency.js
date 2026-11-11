/* Measure the event loop during real local inference over Sofuu's HTTP server.
 *
 * Requires an existing, compatible GGUF with weights. This benchmark never
 * downloads a model or substitutes a model response. The client and server
 * share this process and communicate over a loopback TCP connection; the
 * reported delay therefore includes client, server, inference and GC work.
 *
 *   SOFUU_ALLOW_LOOPBACK=1 SOFUU_LOOP_MODEL=/path/to/model.gguf \
 *     make event-loop-check
 *
 * Optional: SOFUU_LOOP_CONCURRENCY (default 2), SOFUU_LOOP_MAX_TOKENS (24),
 * SOFUU_LOOP_BUDGET_MS (16), SOFUU_LOOP_TIMEOUT_MS (120000), SOFUU_LOOP_PORT.
 * Missing prerequisites exit 2 (blocked), not 0. A failed request or a budget
 * violation exits 1. Historical fake-stream measurements are not comparable.
 */
(function () {
  'use strict';

  var env = typeof process !== 'undefined' && process.env ? process.env : {};
  var modelPath = env.SOFUU_LOOP_MODEL || '';

  function blocked(message) {
    console.error('BLOCKED: event-loop benchmark: ' + message);
    process.exit(2);
  }

  function integer(name, fallback, min, max) {
    var raw = env[name];
    if (raw === undefined || raw === '') return fallback;
    var value = Number(raw);
    if (!/^\d+$/.test(String(raw)) || value < min || value > max || !Number.isInteger(value)) {
      blocked(name + ' must be an integer from ' + min + ' to ' + max);
    }
    return value;
  }

  if (!modelPath) blocked('set SOFUU_LOOP_MODEL to an existing GGUF with real weights');
  if (!/\.gguf$/i.test(modelPath)) blocked('SOFUU_LOOP_MODEL must name a GGUF file with real weights');
  if (!sofuu.fs || typeof sofuu.fs.existsSync !== 'function' || !sofuu.fs.existsSync(modelPath)) {
    blocked('model file does not exist or cannot be accessed: ' + modelPath);
  }
  if (!sofuu.serve || typeof sofuu.serve.start !== 'function' || typeof sofuu.fetch !== 'function') {
    blocked('this Sofuu build does not expose the HTTP server and fetch APIs');
  }
  if (typeof performance === 'undefined' || typeof performance.now !== 'function') {
    blocked('this Sofuu build does not expose a monotonic performance clock');
  }

  var concurrency = integer('SOFUU_LOOP_CONCURRENCY', 2, 1, 32);
  var maxTokens = integer('SOFUU_LOOP_MAX_TOKENS', 24, 1, 256);
  var budgetMs = integer('SOFUU_LOOP_BUDGET_MS', 16, 1, 60000);
  var timeoutMs = integer('SOFUU_LOOP_TIMEOUT_MS', 120000, 1000, 600000);
  var port = integer('SOFUU_LOOP_PORT', 20000 + Math.floor(Math.random() * 30000), 1024, 65535);
  var sampleMs = 5;
  var instance = Math.random().toString(36).slice(2) + Date.now().toString(36);
  var modelId = 'loop-bench-' + instance;
  var token = 'loop-bench-' + instance;
  var base = 'http://127.0.0.1:' + port;
  var timer = null;

  function errorText(error) {
    return String(error && error.message || error);
  }

  function countTextFrames(body) {
    var textFrames = 0;
    var done = false;
    var lines = body.split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].slice(0, 5) !== 'data:') continue;
      var data = lines[i].slice(5).trim();
      if (data === '[DONE]') { done = true; continue; }
      var frame;
      try { frame = JSON.parse(data); }
      catch (e) { throw new Error('invalid SSE JSON frame'); }
      if (frame.error) throw new Error('server returned an inference error');
      if (frame.model && frame.model !== modelId) throw new Error('response came from an unexpected model');
      var choice = frame.choices && frame.choices[0];
      if (choice && choice.delta && typeof choice.delta.content === 'string' && choice.delta.content.length) {
        textFrames++;
      }
    }
    if (!done || !textFrames) throw new Error('stream did not finish with model-generated text');
    return textFrames;
  }

  async function main() {
    try {
      sofuu.serve.start({
        modelPath: modelPath,
        models: [{ id: modelId, path: modelPath }],
        token: token,
        host: '127.0.0.1',
        port: port,
        version: 'real-inference-benchmark',
        brainPath: '',
      });
      // Listen is asynchronous; give it a turn before checking its identity.
      await new Promise(function (resolve) { setTimeout(resolve, 100); });
      var check = await sofuu.fetch(base + '/v1/models', {
        headers: { authorization: 'Bearer ' + token }, timeoutMs: 5000, maxBodyBytes: 65536,
      });
      var listing = await check.json();
      var found = listing && Array.isArray(listing.data) && listing.data.some(function (m) { return m.id === modelId; });
      if (check.status !== 200 || !found) throw new Error('the benchmark server did not identify its configured model');

      var body = JSON.stringify({
        model: modelId,
        messages: [{ role: 'user', content: 'Explain why reliable error messages help software developers in one sentence.' }],
        stream: true,
        max_tokens: maxTokens,
        temperature: 0,
      });
      var samples = [];
      var last = performance.now();
      var started = last;
      timer = setInterval(function () {
        var now = performance.now();
        samples.push(Math.max(0, now - last - sampleMs));
        last = now;
      }, sampleMs);
      var jobs = [];
      for (var i = 0; i < concurrency; i++) {
        jobs.push(sofuu.fetch(base + '/v1/chat/completions', {
          method: 'POST',
          headers: { authorization: 'Bearer ' + token, 'content-type': 'application/json' },
          body: body,
          timeoutMs: timeoutMs,
          maxBodyBytes: 1024 * 1024,
        }).then(async function (response) {
          var text = await response.text();
          if (response.status !== 200) throw new Error('inference request returned HTTP ' + response.status);
          return countTextFrames(text);
        }));
      }
      var frames = await Promise.all(jobs);
      var finished = performance.now();
      // Include the final interval even if the loop was blocked until completion.
      samples.push(Math.max(0, finished - last - sampleMs));
      clearInterval(timer);
      timer = null;
      samples.sort(function (a, b) { return a - b; });
      var total = samples.reduce(function (sum, value) { return sum + value; }, 0);
      var p95 = samples[Math.max(0, Math.ceil(samples.length * 0.95) - 1)];
      var max = samples[samples.length - 1];
      var frameCount = frames.reduce(function (sum, value) { return sum + value; }, 0);
      console.log('event-loop latency: real local inference over loopback HTTP');
      console.log('  model=' + modelPath + ' platform=' + process.platform + '/' + process.arch +
        ' concurrency=' + concurrency + ' max_tokens=' + maxTokens + ' temperature=0');
      console.log('  wall_ms=' + (finished - started).toFixed(3) + ' text_frames=' + frameCount +
        ' samples=' + samples.length + ' sample_interval_ms=' + sampleMs + ' clock=performance.now');
      console.log('  mean_ms=' + (total / samples.length).toFixed(3) + ' p95_ms=' + p95.toFixed(3) +
        ' max_ms=' + max.toFixed(3) + ' budget_ms=' + budgetMs);
      if (max > budgetMs) {
        console.error('FAIL: observed loop delay exceeded the configured budget');
        process.exit(1);
      }
      console.log('PASS: all requests produced text and the observed delay stayed within budget');
      process.exit(0);
    } catch (error) {
      if (timer !== null) clearInterval(timer);
      console.error('FAIL: real inference benchmark did not complete: ' + errorText(error));
      process.exit(1);
    }
  }

  main();
})();
