// evaluators.js — `sofuu.evaluators`: the standardized deterministic rule
// contract (PLAN-AUTONOMOUS-REACTIVE-RUNTIME Phase 3).
//
// Many AI applications let users drop their own rules/indicators/heuristics
// into a directory (a static analyzer, a threshold scorer, a log filter).
// This driver turns that convention into one contract:
//
//   export default {
//     id: 'rsi_reversion',                    // required, stable
//     name: 'RSI reversion',                  // optional display name
//     weight: 1,                              // optional composite weight
//     evaluate(input) {                       // required, may be async
//       return { signal, score, confidence, metrics, reason };
//     }
//   };
//
// `sofuu.evaluators.loadDir(dir)` imports every *.js module in sorted order
// and validates the contract. A malformed file is REPORTED and skipped — it
// can never take down a 24/7 loop. `score(input, evaluators, {threshold})`
// runs each evaluator in isolation and produces the normalized envelope:
//
//   { id, name, signal, score: [-100,100], confidence: [0,1],
//     metrics: {}, reason }
//
// plus a confidence-weighted composite and a deterministic `triggered` flag
// that can be wired straight into a pre-LLM `gate(ctx)`.
//
// Errors (`RULES.md §3.5`): E_EVALUATOR_CONTRACT for a module/file that does
// not satisfy the contract. Bad *one* evaluator is data in `errors[]`; bad
// arguments to loadDir/score are thrown.

(function () {
  'use strict';
  var global = globalThis;
  if (!global.sofuu) global.sofuu = {};

  function makeError(code, msg, details) {
    var err = new Error(msg);
    err.name = 'SofuuError';
    err.code = code || 'E_EVALUATOR_CONTRACT';
    err.rule = 'RULES.md §3.5';
    if (details) err.details = details;
    return err;
  }

  function cwd() {
    try {
      if (typeof process !== 'undefined' && process && typeof process.cwd === 'function') {
        return String(process.cwd());
      }
    } catch (e) { /* an embedded host without a cwd uses '.' */ }
    return '.';
  }

  function absPath(p) {
    var s = String(p == null ? '' : p);
    if (!s) return s;
    if (s.charAt(0) === '/') return s;
    if (/^[A-Za-z]:[\\/]/.test(s)) return s;
    var base = cwd().replace(/[\\/]+$/, '');
    return base + '/' + s.replace(/^\.\//, '');
  }

  function baseName(p) {
    var s = String(p == null ? '' : p);
    var i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return i >= 0 ? s.slice(i + 1) : s;
  }

  function dirName(p) {
    var s = String(p == null ? '' : p);
    var i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return i > 0 ? s.slice(0, i) : '.';
  }

  function clamp(n, lo, hi) { return n < lo ? lo : (n > hi ? hi : n); }

  function finiteOr(v, fallback) {
    var n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function round(n, places) {
    var f = Math.pow(10, places == null ? 2 : places);
    return Math.round(n * f) / f;
  }

  function timeoutError(ms, id) {
    return makeError('E_EVALUATOR_CONTRACT',
      'evaluator' + (id ? ' \'' + id + '\'' : '') + ' exceeded its ' + ms + 'ms budget',
      { id: id || '', timeoutMs: ms });
  }

  function withTimeout(promise, ms, id) {
    if (!(ms > 0)) return Promise.resolve(promise);
    return new Promise(function (resolve, reject) {
      var settled = false;
      var t = setTimeout(function () {
        if (settled) return;
        settled = true;
        reject(timeoutError(ms, id));
      }, ms);
      Promise.resolve(promise).then(function (v) {
        if (settled) return;
        settled = true;
        clearTimeout(t);
        resolve(v);
      }, function (e) {
        if (settled) return;
        settled = true;
        clearTimeout(t);
        reject(e);
      });
    });
  }

  /* ── contract ─────────────────────────────────────────────────────── */

  function describeError(e, code) {
    return {
      code: (e && e.code) || code || 'E_EVALUATOR_CONTRACT',
      message: String((e && e.message) || e || 'unknown evaluator failure')
    };
  }

  /**
   * Validate one candidate object (default export, `evaluator`, `algorithm`,
   * or the module namespace itself). Returns null when invalid.
   */
  function candidateOf(mod) {
    if (!mod) return null;
    if (typeof mod === 'object' || typeof mod === 'function') {
      var cands = [mod.default, mod.evaluator, mod.algorithm, mod];
      for (var i = 0; i < cands.length; i++) {
        var c = cands[i];
        if (!c) continue;
        if (typeof c === 'function' && typeof c.evaluate !== 'function') {
          // A factory returning the evaluator is convenient and explicit.
          continue;
        }
        if (typeof c === 'object' && typeof c.id === 'string' &&
            c.id.trim() && typeof c.evaluate === 'function') {
          return c;
        }
      }
    }
    return null;
  }

  function normalizeEvaluator(raw, file) {
    return {
      id: String(raw.id).trim(),
      name: String(raw.name || raw.id),
      // Optional composite weight; 1 is the neutral default. A negative or
      // non-finite weight is treated as 1 rather than silently inverting.
      weight: finiteOr(raw.weight, 1) > 0 ? finiteOr(raw.weight, 1) : 1,
      file: file || '',
      evaluate: raw.evaluate
    };
  }

  /**
   * The normalized envelope. Every consumer — gate, UI, log — reads this
   * shape, so an evaluator cannot leak arbitrary fields into the contract.
   * Score is clamped to [-100, 100], confidence to [0, 1]; missing values
   * take the neutral defaults (0 / 0.5), never NaN.
   */
  function normalize(output, evaluator) {
    var out = (output && typeof output === 'object') ? output : {};
    var score = finiteOr(out.score, 0);
    var conf = finiteOr(out.confidence, 0.5);
    var metrics = (out.metrics && typeof out.metrics === 'object' && !Array.isArray(out.metrics))
      ? out.metrics : {};
    return {
      id: String(evaluator && evaluator.id || out.id || ''),
      name: String((out.name || (evaluator && evaluator.name) || (evaluator && evaluator.id) || '')),
      signal: String(out.signal || 'NEUTRAL'),
      score: round(clamp(score, -100, 100), 2),
      confidence: round(clamp(conf, 0, 1), 3),
      metrics: metrics,
      reason: String(out.reason == null ? '' : out.reason)
    };
  }

  /* ── loadDir ──────────────────────────────────────────────────────── */

  async function loadDir(dirPath, opts) {
    opts = opts || {};
    if (dirPath === undefined || dirPath === null || String(dirPath).trim() === '') {
      throw makeError('E_EVALUATOR_CONTRACT', 'evaluators.loadDir: dirPath is required');
    }
    var dir = String(dirPath);
    var timeoutMs = finiteOr(opts.timeoutMs, 0);
    var extensions = Array.isArray(opts.extensions) && opts.extensions.length
      ? opts.extensions.map(function (e) { return String(e).replace(/^\*?/, ''); })
      : ['.js', '.mjs'];
    if (!global.sofuu.fs || typeof global.sofuu.fs.readdir !== 'function') {
      throw makeError('E_EVALUATOR_CONTRACT', 'evaluators.loadDir: sofuu.fs is unavailable in this build');
    }

    var names;
    try {
      names = await global.sofuu.fs.readdir(dir);
    } catch (e) {
      throw makeError('E_EVALUATOR_CONTRACT',
        'evaluators.loadDir: cannot read ' + dir + ': ' + String((e && e.message) || e));
    }
    if (!Array.isArray(names)) names = [];
    names = names.map(function (n) { return String(n); })
      .filter(function (n) {
        for (var i = 0; i < extensions.length; i++) {
          if (n.length >= extensions[i].length &&
              n.slice(-extensions[i].length) === extensions[i]) return true;
        }
        return false;
      })
      .sort();

    var evaluators = [];
    var errors = [];
    var seen = Object.create(null);
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      var full = absPath(dir.replace(/[\\/]+$/, '') + '/' + name);
      var mod;
      /* R7: the ESM loader reads the file directly, so this preflight is the
       * one place fs.read / fs.deny is enforced for a hot-loaded rule. A
       * policy denial is reported as a contract error and the file is never
       * imported. */
      try {
        if (global.sofuu.fs && typeof global.sofuu.fs.readFile === 'function') {
          await global.sofuu.fs.readFile(full, 'utf8');
        }
      } catch (ePol) {
        errors.push({ file: full, id: '', error: describeError(ePol, 'E_DENIED') });
        continue;
      }
      try {
        mod = await import(full);
      } catch (eImp) {
        errors.push({ file: full, id: '', error: describeError(eImp) });
        continue;
      }
      var cand = candidateOf(mod);
      if (!cand) {
        errors.push({
          file: full, id: '',
          error: {
            code: 'E_EVALUATOR_CONTRACT',
            message: 'module does not export { id: string, evaluate: function } ' +
                     '(accepted: default, evaluator, algorithm, or the module namespace)'
          }
        });
        continue;
      }
      var ev = normalizeEvaluator(cand, full);
      if (seen[ev.id]) {
        errors.push({
          file: full, id: ev.id,
          error: { code: 'E_EVALUATOR_CONTRACT', message: 'duplicate evaluator id \'' + ev.id + '\'' }
        });
        continue;
      }
      seen[ev.id] = true;
      evaluators.push(ev);
    }
    return { dir: absPath(dir), evaluators: evaluators, errors: errors, timeoutMs: timeoutMs };
  }

  /* ── score ────────────────────────────────────────────────────────── */

  /**
   * Evaluate every evaluator against `input` in isolation. One throwing,
   * malformed or slow evaluator contributes a neutral ERROR envelope and an
   * entry in `errors[]`; it never rejects the whole call.
   */
  async function score(input, evaluators, opts) {
    opts = opts || {};
    if (!Array.isArray(evaluators)) {
      throw makeError('E_EVALUATOR_CONTRACT', 'evaluators.score: evaluators must be an array');
    }
    var timeoutMs = finiteOr(opts.timeoutMs, 0);
    var threshold = finiteOr(opts.threshold, 40);
    var includeNeutral = opts.includeNeutral !== false;
    var signals = [];
    var errors = [];

    for (var i = 0; i < evaluators.length; i++) {
      var ev = evaluators[i];
      if (!ev || typeof ev.evaluate !== 'function' || typeof ev.id !== 'string') {
        errors.push({
          id: String(ev && ev.id || ''),
          error: { code: 'E_EVALUATOR_CONTRACT', message: 'evaluator is missing id/evaluate' }
        });
        continue;
      }
      var envelope;
      try {
        var raw = await withTimeout(
          Promise.resolve(ev.evaluate(input)),
          timeoutMs, ev.id);
        envelope = normalize(raw, ev);
        if (envelope.signal === 'NEUTRAL' && !includeNeutral) continue;
      } catch (eEval) {
        errors.push({ id: ev.id, error: describeError(eEval) });
        envelope = {
          id: ev.id, name: ev.name || ev.id, signal: 'ERROR',
          score: 0, confidence: 0, metrics: {}, reason: String((eEval && eEval.message) || eEval || '')
        };
        envelope.error = describeError(eEval);
      }
      envelope.weight = finiteOr(ev.weight, 1) > 0 ? finiteOr(ev.weight, 1) : 1;
      signals.push(envelope);
    }

    var weighted = 0;
    var weightTotal = 0;
    var maxAbs = 0;
    var totalConfidence = 0;
    for (var j = 0; j < signals.length; j++) {
      var s = signals[j];
      var w = s.confidence * s.weight;
      weighted += s.score * w;
      weightTotal += w;
      totalConfidence += s.confidence;
      var a = Math.abs(s.score);
      if (a > maxAbs) maxAbs = a;
    }
    var composite = weightTotal > 0 ? (weighted / weightTotal) : 0;
    var triggered = signals.length > 0 &&
      (Math.abs(composite) >= threshold || maxAbs >= threshold);
    return {
      signals: signals,
      errors: errors,
      compositeScore: round(composite, 2),
      maxAbsScore: round(maxAbs, 2),
      confidence: signals.length ? round(totalConfidence / signals.length, 3) : 0,
      threshold: threshold,
      triggered: triggered
    };
  }

  var api = {
    loadDir: loadDir,
    score: score,
    normalize: normalize,
    candidateOf: candidateOf,
    // Exposed for apps that want the same contract validation on a
    // programmatically-built evaluator list.
    validate: function (list) {
      if (!Array.isArray(list)) {
        throw makeError('E_EVALUATOR_CONTRACT', 'evaluators.validate: expected an array');
      }
      var out = [];
      for (var i = 0; i < list.length; i++) {
        var ev = normalizeEvaluator(list[i] || {}, '');
        if (!ev.id || typeof ev.evaluate !== 'function') {
          throw makeError('E_EVALUATOR_CONTRACT',
            'evaluators.validate: entry ' + i + ' lacks id/evaluate');
        }
        out.push(ev);
      }
      return out;
    }
  };

  global.sofuu.evaluators = api;
  if (!global.Sofuu) global.Sofuu = {};
  global.Sofuu.evaluators = api;
})();
