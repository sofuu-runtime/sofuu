/* deploy/pi5/ambient/agent.js — the M-R3 ambient-device program.
 *
 * One program, two deployments:
 *
 *   development   sofuu run deploy/pi5/ambient/agent.js --stats
 *                 (brain opened from SOFUU_DEVICE_BRAIN, or ./device.brain.qtsq)
 *
 *   production    ./ambient --stats
 *                 (a `sofuu compile --embed-brain` artifact; the brain is the
 *                  seed the artifact carries, materialised beside it on first
 *                  launch and never overwritten afterwards)
 *
 * It is deliberately offline and deterministic: no model, no fetch, no shell,
 * no tools. Its one side effect is a record appended to its own brain, which
 * is what makes the deployment testable (seed extraction, recall, WAL
 * recovery) without a vehicle, a sensor HAT, or a network.
 *
 * The sensor is the Pi's SoC thermal zone, read through sofuu.fs and therefore
 * through the R7 policy. On a machine without it (a Mac, a container) the
 * record says the sensor is unavailable rather than inventing a value.
 *
 * Commands (sofuu.args in an artifact, process.argv under `sofuu run`):
 *   --once                    append one reading and exit
 *   --reading <value>         append a caller-supplied value and exit
 *   --recall <query> [k]      read-only: print the k best matches
 *   --stats                   read-only: print path, dim, record count
 *   --help
 *   (no command)              serve: one reading now, then every interval
 */
'use strict';

(function () {
  var S = globalThis.sofuu;
  if (!S || !S.memory) {
    console.error(JSON.stringify({ level: 'error', code: 'E_RUNTIME',
      message: 'this program needs the Sofuu runtime (run it with sofuu)' }));
    if (typeof process !== 'undefined' && process.exit) process.exit(3);
    throw new Error('no sofuu runtime');
  }

  var ARGV = (S.args && Array.isArray(S.args))
    ? S.args
    : ((typeof process !== 'undefined' && process.argv) ? process.argv.slice(2) : []);
  /* Under `sofuu run`, process.argv is [sofuu, script, ...args]; a preceding
   * CLI flag moves the script into an earlier slot, so drop a script-looking
   * first argument rather than mistaking it for a command. */
  if (!(S.args && Array.isArray(S.args)) && ARGV.length &&
      /\.(c|m)?[jt]s$/.test(ARGV[0])) {
    ARGV = ARGV.slice(1);
  }
  var ENV = (typeof process !== 'undefined' && process.env) ? process.env : {};

  function log(event, fields) {
    var row = { ts: new Date().toISOString(), event: event };
    if (fields) for (var k in fields) row[k] = fields[k];
    console.log(JSON.stringify(row));
  }

  function die(code, message, exit) {
    log('fatal', { code: code, message: message });
    if (typeof process !== 'undefined' && process.exit) process.exit(exit || 4);
    throw new Error(code + ': ' + message);
  }

  /* ── policy ───────────────────────────────────────────────────────
   * `sofuu.policy.load()` merges ~/.sofuu/policy.json with
   * <cwd>/.sofuu/policy.json. It is not automatic for `sofuu run`, so the
   * device asks for it explicitly and refuses to serve on a malformed file:
   * the runtime's own rule is "invalid leaves the previous policy", and the
   * previous policy of a fresh process is empty (allow-all). On a device that
   * is the wrong way to fail. */
  var policy = null;
  if (S.policy && typeof S.policy.load === 'function') {
    var raw = S.policy.load();
    try {
      policy = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (e) {
      die('E_POLICY', 'sofuu.policy.load() did not answer JSON: ' + e, 4);
    }
    if (!policy || policy.ok !== true) {
      die('E_POLICY', 'policy did not load: ' + JSON.stringify(policy), 4);
    }
    if (!policy.sources || policy.sources.length === 0) {
      log('warn', { code: 'E_POLICY_ABSENT',
        message: 'no .sofuu/policy.json found — running with the runtime default (no R7 rules)' });
    }
  }

  function netDecision() {
    if (!S.policy || typeof S.policy.evaluate !== 'function') return null;
    try {
      var out = S.policy.evaluate(JSON.stringify({ op: 'net.fetch', host: 'example.com' }));
      return typeof out === 'string' ? JSON.parse(out) : out;
    } catch (e) {
      return { decision: 'deny', rule: 'probe-failed', reason: String(e) };
    }
  }

  /* ── brain ────────────────────────────────────────────────────────
   * The compiled shape installs S.memory.embedded; the development shape
   * opens a file. A compiled artifact whose seed failed to mount has neither,
   * and that must be loud: silently creating a fresh empty brain beside it
   * would look like success while losing every remembered record. */
  var brain = null;
  var brainPath = '';
  var brainDim = 0;
  var embedded = (S.memory && S.memory.embedded && typeof S.memory.embedded.open === 'function')
    ? S.memory.embedded : null;

  if (embedded) {
    brain = embedded.open();
    brainPath = embedded.path;
    brainDim = embedded.dim;
  } else {
    brainPath = ENV.SOFUU_DEVICE_BRAIN ||
      ((typeof process !== 'undefined' && process.cwd) ? process.cwd() + '/device.brain.qtsq'
                                                       : './device.brain.qtsq');
    brainDim = parseInt(ENV.SOFUU_DEVICE_DIM || '768', 10) || 768;
    brain = S.memory.open(brainPath, brainDim);
  }
  if (!brain || typeof brain.remember !== 'function') {
    die('E_NO_SEED', 'no brain: neither an embedded seed nor an openable file at ' + brainPath, 3);
  }

  var DEVICE_NAME = ENV.SOFUU_DEVICE_NAME || 'ambient-01';
  var DEVICE_SITE = ENV.SOFUU_DEVICE_SITE || 'bench';
  var INTERVAL_MS = parseInt(ENV.SOFUU_READING_INTERVAL_MS || '60000', 10) || 60000;

  /* First launch: put one durable record in an empty brain. This is the
   * record a later `--recall seed` proves survived the deploy/restart. */
  function seedIfEmpty() {
    if (brain.count() > 0) return false;
    brain.remember(
      'device profile: name=' + DEVICE_NAME + ' site=' + DEVICE_SITE +
      ' class=ambient deployed=' + new Date().toISOString(),
      'system', 0, 'seed');
    brain.flush();
    log('seed', { path: brainPath, dim: brainDim, name: DEVICE_NAME, site: DEVICE_SITE });
    return true;
  }

  /* ── sensor ───────────────────────────────────────────────────────
   * /sys/class/thermal/thermal_zone0/temp is millidegrees C. The read is a
   * real R7 fs.read: the policy allowlist in .sofuu/policy.json names the
   * directory, and a policy that omits it returns E_PERMISSION_DENIED here —
   * which is reported as "sensor unavailable", never papered over. */
  function readSocTemp() {
    try {
      if (!S.fs || typeof S.fs.readFileSync !== 'function') return null;
      var buf = S.fs.readFileSync('/sys/class/thermal/thermal_zone0/temp');
      var text = new TextDecoder().decode(buf).trim();
      var milli = parseInt(text, 10);
      if (!isFinite(milli)) return null;
      return Math.round((milli / 1000) * 10) / 10;
    } catch (e) {
      return null;
    }
  }

  function appendReading(value, source) {
    var c = (value === null || value === undefined) ? 'unavailable' : String(value);
    var text = 'reading soc_temp_c=' + c + ' at ' + new Date().toISOString() +
      ' device=' + DEVICE_NAME;
    brain.remember(text, 'device', 0, source || 'sensor');
    brain.flush();
    log('reading', { sensor: 'soc_temp_c', value: value, path: brainPath });
    return text;
  }

  /* ── commands ────────────────────────────────────────────────────── */
  function usage() {
    console.log('ambient device agent\n' +
      '  --once                  append one reading and exit\n' +
      '  --reading <value>       append a supplied value and exit\n' +
      '  --recall <query> [k]    read-only: print the k best matches\n' +
      '  --stats                 read-only: path, dim, record count\n' +
      '  --help\n' +
      '  (no command)            serve: one reading now, then every ' + INTERVAL_MS + ' ms');
  }

  function stats() {
    log('stats', {
      path: brainPath,
      dim: brainDim,
      embedded: !!embedded,
      records: brain.count(),
      policy_sources: policy ? policy.sources : [],
      net: netDecision()
    });
  }

  function recall(query, k) {
    var hits = brain.recall(query, k || 3);
    console.log(JSON.stringify({
      ts: new Date().toISOString(),
      event: 'recall',
      query: query,
      hits: (hits || []).map(function (h) {
        return {
          text: h.text,
          similarity: (typeof h.similarity === 'number') ? Math.round(h.similarity * 10000) / 10000 : h.similarity,
          role: h.role
        };
      })
    }, null, 2));
  }

  var cmd = ARGV[0] || '';

  if (cmd === '--help' || cmd === '-h' || cmd === 'help') {
    usage();
    return;
  }

  if (cmd === '--stats') {
    stats();
    return;
  }

  if (cmd === '--recall') {
    var q = ARGV[1];
    if (!q) die('E_USAGE', '--recall needs a query', 2);
    recall(q, parseInt(ARGV[2] || '3', 10) || 3);
    return;
  }

  /* Read-only commands above never write; --stats on a fresh brain must not
   * seed it, or the deployment check could not tell "the service seeded this"
   * from "the check did". */
  seedIfEmpty();

  if (cmd === '--reading') {
    var v = ARGV[1];
    if (v === undefined) die('E_USAGE', '--reading needs a value', 2);
    appendReading(v, 'manual');
    return;
  }

  if (cmd === '--once') {
    appendReading(readSocTemp(), 'sensor');
    return;
  }

  if (cmd !== '') {
    usage();
    die('E_USAGE', 'unknown command: ' + cmd, 2);
  }

  /* Serve. One reading immediately, then on an interval; systemd owns the
   * lifetime. Every record is flushed on its own, so a SIGKILL at any point
   * loses at most the record being written — which is exactly what the WAL
   * recovery test in verify_deploy.sh exercises. */
  log('start', {
    path: brainPath,
    dim: brainDim,
    embedded: !!embedded,
    interval_ms: INTERVAL_MS,
    policy_sources: policy ? policy.sources : [],
    net: netDecision(),
    records: brain.count()
  });
  appendReading(readSocTemp(), 'sensor');
  setInterval(function () {
    try {
      appendReading(readSocTemp(), 'sensor');
    } catch (e) {
      log('error', { code: 'E_WRITE', message: String(e) });
    }
  }, INTERVAL_MS);
})();
