/* deploy/pi5/ambient/seed.js — create the brain that the artifact carries.
 *
 * Run it AS THE SERVICE USER, on the device, so the seed is encrypted with the
 * same machine key the service later reads ($HOME/.sofuu/brain.key):
 *
 *   sudo -u sofuu -H /usr/local/bin/sofuu run /usr/local/lib/sofuu/ambient/seed.js \
 *        /var/lib/sofuu/seed.brain.qtsq
 *
 * `sofuu compile --embed-brain` packs the manifest AND its .d/ segments, so a
 * brain with zero records is refused ("carries no records") — this writes two.
 *
 *   seed.js <path.qtsq> [dim] [name] [site]
 */
'use strict';

(function () {
  var S = globalThis.sofuu;
  if (!S || !S.memory) {
    console.error('seed.js: needs the Sofuu runtime with the QTSQ codec');
    if (typeof process !== 'undefined' && process.exit) process.exit(3);
    throw new Error('no runtime');
  }

  var ARGV = (S.args && Array.isArray(S.args))
    ? S.args
    : ((typeof process !== 'undefined' && process.argv) ? process.argv.slice(2) : []);
  if (!(S.args && Array.isArray(S.args)) && ARGV.length &&
      /\.(c|m)?[jt]s$/.test(ARGV[0])) {
    ARGV = ARGV.slice(1);
  }

  var path = ARGV[0];
  if (!path) {
    console.error('usage: seed.js <path.qtsq> [dim] [name] [site]');
    if (typeof process !== 'undefined' && process.exit) process.exit(2);
    throw new Error('missing path');
  }
  var dim = parseInt(ARGV[1] || '768', 10) || 768;
  var name = ARGV[2] || 'ambient-01';
  var site = ARGV[3] || 'bench';

  var brain = S.memory.open(path, dim);
  var before = brain.count();
  if (before > 0) {
    console.log(JSON.stringify({ ok: true, path: path, dim: dim, records: before,
      note: 'already seeded — left untouched' }));
    return;
  }

  brain.remember(
    'device profile: name=' + name + ' site=' + site + ' class=ambient' +
    ' seeded=' + new Date().toISOString(),
    'system', 0, 'seed');
  brain.remember(
    'device rule: report soc_temp_c every interval; no actuation in v1 ' +
    '(hardware classes arrive with the shared R7 safety core)',
    'system', 0, 'seed');
  brain.flush();

  console.log(JSON.stringify({ ok: true, path: path, dim: dim, records: brain.count() }));
})();
