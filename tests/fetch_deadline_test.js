// tests/fetch_deadline_test.js — js-8 (AUDIT-2026-09-07) regression.
//
// fetch's timeoutMs option must become curl's own CURLOPT_TIMEOUT_MS:
// a request to an endpoint that accepts but never answers must die near
// the caller's deadline (promise rejection) instead of hanging until the
// global 120s default. A watchdog fails fast at 6s if the option is
// ignored, so the pre-fix binary cannot hang the suite.

var results = { passed: 0, failed: 0 };
function assert(label, cond) {
  if (cond) { console.log('PASS  ' + label); results.passed++; }
  else { console.log('FAIL  ' + label); results.failed++; }
}
function msg(e) { return String((e && e.message) || e); }

var BH_PORT = 13777;
var OK_PORT = 13778;

function sleep(ms) {
  return new Promise(function (res) { setTimeout(res, ms); });
}
function watchdog(ms, tag) {
  return new Promise(function (res) { setTimeout(function () { res(tag); }, ms); });
}

async function main() {
  assert('sofuu.fetch is a function', typeof sofuu.fetch === 'function');

  // Black-hole endpoint: accepts the request and never responds. No
  // subprocess needed — a handler that never calls res.end() holds the
  // connection open silently, which is exactly the stall js-8 fixes.
  var blackhole = sofuu.http.createServer(function (req, res) {
    /* never respond */
  });
  blackhole.listen(BH_PORT, '127.0.0.1');
  await sleep(250);

  // (a) Control — WITHOUT timeoutMs the fetch must still be pending at
  // 3s (endpoint stalls). A fast rejection here means the listener is
  // not really black-holing, which would invalidate (b).
  var controlAlive = false, controlErr = '';
  try {
    var w1 = await Promise.race([
      sofuu.fetch('http://127.0.0.1:13777/never', {}),
      watchdog(3000, 'ALIVE'),
    ]);
    controlAlive = (w1 === 'ALIVE');
    if (!controlAlive) controlErr = 'fetch settled early';
  } catch (e1) {
    controlErr = msg(e1);
  }
  assert('control: stalled endpoint leaves fetch pending at 3s', controlAlive);
  if (!controlAlive && controlErr) console.log('      (' + controlErr + ')');

  // (b) The fix — timeoutMs 700 must kill the fetch near the deadline.
  var t0 = Date.now();
  var deadlineHonored = false, elapsed = 0, deadlineErr = '';
  try {
    var w2 = await Promise.race([
      sofuu.fetch('http://127.0.0.1:13777/never', { timeoutMs: 700 }),
      watchdog(6000, 'NO-DEADLINE'),
    ]);
    deadlineErr = (w2 === 'NO-DEADLINE')
      ? 'timeoutMs ignored — still pending at 6s'
      : 'fetch resolved instead of rejecting';
  } catch (e2) {
    elapsed = Date.now() - t0;
    deadlineHonored = elapsed >= 400 && elapsed <= 4000;
    if (!deadlineHonored) deadlineErr = 'rejected but at ' + elapsed + 'ms (expected ~700)';
  }
  assert('timeoutMs honored: stalled fetch rejects near the 700ms deadline', deadlineHonored);
  if (deadlineErr) console.log('      (' + deadlineErr + ')');
  if (deadlineHonored) console.log('      rejected after ' + elapsed + 'ms');

  // (c) Fast endpoint with a generous timeoutMs still completes.
  var server = sofuu.http.createServer(function (req, res) {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('fine');
  });
  server.listen(OK_PORT, '127.0.0.1');
  await sleep(250);
  try {
    var r = await Promise.race([
      sofuu.fetch('http://127.0.0.1:13778/health', { timeoutMs: 5000 }),
      watchdog(9000, 'STUCK'),
    ]);
    var fast = false;
    if (r !== 'STUCK') {
      var txt = await r.text();
      fast = r.status === 200 && txt === 'fine';
    }
    assert('fast endpoint with timeoutMs still completes', fast);
  } catch (e3) {
    assert('fast endpoint with timeoutMs still completes', false);
    console.log('      (' + msg(e3) + ')');
  }
  if (server.stop) server.stop();
  if (server.close) server.close();
  if (blackhole.stop) blackhole.stop();
  if (blackhole.close) blackhole.close();

  console.log('---');
  console.log('passed=' + results.passed + ' failed=' + results.failed);
  console.log('FETCH-DEADLINE TEST: ' + (results.failed === 0 ? 'ALL PASSED' : 'FAILED'));
  process.exit(results.failed === 0 ? 0 : 1);
}

main().catch(function (e) {
  console.log('HARNESS ERROR: ' + msg(e));
  try { process.exit(1); } catch (eX) {}
});
