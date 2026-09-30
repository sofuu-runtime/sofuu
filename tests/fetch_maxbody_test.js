// tests/fetch_maxbody_test.js — js-9 (AUDIT-2026-09-07) regression.
//
// fetch's maxBodyBytes option must abort the transfer once the buffered
// body would cross the cap: the promise rejects naming the cap instead of
// buffering up to the global 256MB hard cap. Also covers web.open, whose
// whole-body buffering preceded the maxChars truncation — SOFUU_WEB_
// ALLOW_INTERNAL=1 lets the harness point web.open at the loopback
// fixture (the opt-out is read at call time).

process.env.SOFUU_WEB_ALLOW_INTERNAL = '1';

var results = { passed: 0, failed: 0 };
function assert(label, cond) {
  if (cond) { console.log('PASS  ' + label); results.passed++; }
  else { console.log('FAIL  ' + label); results.failed++; }
}
function msg(e) { return String((e && e.message) || e); }

var PORT = 13801;
var BIG = new Array(200 * 1024 + 1).join('x'); // 200 KB constant body
var HUGE_PAGE = '<html><body>' + new Array(3 * 1024 * 1024 + 1).join('y') + '</body></html>';

var server = sofuu.http.createServer(function (req, res) {
  // Constant replies only — nothing from the request is echoed back.
  if (req.url === '/big') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end(BIG);
  } else if (req.url === '/huge') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(HUGE_PAGE);
  } else {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
  }
});
server.listen(PORT, '127.0.0.1');

function sleep(ms) {
  return new Promise(function (res) { setTimeout(res, ms); });
}

async function main() {
  await sleep(250); // let the listener bind

  var BASE = 'http://127.0.0.1:13801';

  // (a) Over-cap transfer aborts and names the cap.
  try {
    var r1 = await sofuu.fetch(BASE + '/big', { maxBodyBytes: 1000 });
    var t1 = await r1.text();
    assert('maxBodyBytes aborts an over-cap transfer', false);
    console.log('      (resolved with ' + t1.length + ' chars — cap ignored)');
  } catch (e1) {
    var m1 = msg(e1);
    assert('maxBodyBytes aborts an over-cap transfer', m1.indexOf('maxBodyBytes') >= 0);
    if (m1.indexOf('maxBodyBytes') < 0) console.log('      (wrong rejection: ' + m1 + ')');
  }

  // (b) Same endpoint, no cap — 200KB buffers fine (default unchanged).
  try {
    var r2 = await sofuu.fetch(BASE + '/big');
    var t2 = await r2.text();
    assert('no cap: 200KB body still buffers fully', r2.status === 200 && t2.length === 200 * 1024);
  } catch (e2) {
    assert('no cap: 200KB body still buffers fully', false);
    console.log('      (' + msg(e2) + ')');
  }

  // (c) Under-cap body passes the cap untouched.
  try {
    var r3 = await sofuu.fetch(BASE + '/small', { maxBodyBytes: 1000 });
    var t3 = await r3.text();
    assert('under-cap body passes through', r3.status === 200 && t3 === 'ok');
  } catch (e3) {
    assert('under-cap body passes through', false);
    console.log('      (' + msg(e3) + ')');
  }

  // (d) web.open: 3MB page vs the max(2MB, maxChars*8) transfer cap —
  // must error instead of buffering the whole page.
  var webAvailable = (typeof sofuu.web !== 'undefined') && !!sofuu.web._loaded;
  assert('sofuu.web loaded in run context', webAvailable);
  if (webAvailable) {
    try {
      var out = await sofuu.web.open(BASE + '/huge', { maxChars: 100 });
      assert('web.open errors on a page over the transfer cap', false);
      console.log('      (resolved, text ' + out.text.length + ' chars — cap ignored)');
    } catch (e4) {
      var m4 = msg(e4);
      assert('web.open errors on a page over the transfer cap', m4.indexOf('maxBodyBytes') >= 0);
      if (m4.indexOf('maxBodyBytes') < 0) console.log('      (wrong rejection: ' + m4 + ')');
    }

    // (e) web.open on a small page still works end to end.
    try {
      var out5 = await sofuu.web.open(BASE + '/small', { maxChars: 100 });
      assert('web.open small page still works', out5.status === 200 && out5.text === 'ok');
    } catch (e5) {
      assert('web.open small page still works', false);
      console.log('      (' + msg(e5) + ')');
    }
  }

  if (server.stop) server.stop();
  if (server.close) server.close();

  console.log('---');
  console.log('passed=' + results.passed + ' failed=' + results.failed);
  console.log('FETCH-MAXBODY TEST: ' + (results.failed === 0 ? 'ALL PASSED' : 'FAILED'));
  process.exit(results.failed === 0 ? 0 : 1);
}

main().catch(function (e) {
  console.log('HARNESS ERROR: ' + msg(e));
  try { process.exit(1); } catch (eX) {}
});
