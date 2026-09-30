// tests/rlm_latch_test.js — js-10 (AUDIT-2026-09-07) regression.
//
// Two CONCURRENT sofuu.rlm.query episodes: the first one to tear down
// must NOT clear the shared globalThis.__rlm_aborted latch while a
// sibling episode is still registered — the old unconditional clear in
// the finally block let whichever episode exited first wipe the Esc
// signal before a slower sibling's watcher/boundary check saw it.
//
// Determinism: the watcher only COPIES the latch into ep.aborted; only
// the finally clears it. So asserting the latch value immediately after
// the first episode resolves is race-free. Timing: episode A starts at
// t≈0 (watcher ticks ~100/200/300…), episode B at t≈50 (ticks
// ~150/250/350…). The latch lands at t≈210 — just after A's tick, so
// B's tick (250) kills B first, B resolves at ~260 with A still live.
//   - Fixed:    B's teardown leaves the latch set → A aborts at its 300
//               tick → A resolves 'aborted' fast; A (last out) re-arms
//               the latch to false (P3-13 stillborn guard preserved).
//   - Buggy:    B's teardown clears at ~265 → A never sees the signal →
//               A drips to its wall budget → both assertions fail.

const NEEDLE_MARKER = 'RLM-LATCH';

let failures = 0;
function check(name, cond) {
  if (cond) console.log('PASS ' + name);
  else { failures++; console.log('FAIL ' + name); }
}

function replyJson(res, content) {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.send(JSON.stringify({
    choices: [{ message: { role: 'assistant', content: content } }],
    usage: { prompt_tokens: 1, completion_tokens: 1 }
  }));
}

/* Spread one action script across n SSE chunks (one every perChunkMs).
 * The concatenated content is a valid ```js action block, so an episode
 * that is NOT aborted mid-drip proceeds to the next round instead of
 * erroring — that is what keeps the buggy path alive to its wall cap. */
function sseDripScript(res, n, perChunkMs, script) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  const per = Math.max(1, Math.ceil(script.length / n));
  const pieces = [];
  for (let i = 0; i < script.length; i += per) pieces.push(script.slice(i, i + per));
  let i = 0;
  const t = setInterval(() => {
    if (i >= pieces.length) {
      clearInterval(t);
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }
    res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: pieces[i] } }] }) + '\n\n');
    i++;
  }, perChunkMs);
}

/* Minimal OpenAI-compatible mock with a scripted policy. */
function handler(req, res) {
  let body;
  try { body = JSON.parse(req.body || '{}'); }
  catch (e) { return replyJson(res, 'mock: bad json'); }
  const msgs = body.messages || [];
  const last = String(msgs.length ? (msgs[msgs.length - 1].content || '') : '');
  const isMain = msgs.length > 0 && String(msgs[0].role) === 'system';
  if (!isMain) return replyJson(res, 'sub-answer: nothing here');

  if (body.stream && last.indexOf('LATCH-A') >= 0)
    return sseDripScript(res, 8, 150, '```js\n"AWORK r1"\n```');
  if (body.stream && last.indexOf('LATCH-B') >= 0)
    return sseDripScript(res, 8, 100, '```js\n"BWORK r1"\n```');
  if (last.indexOf('AWORK') >= 0)   /* bug path: A keeps cycling slow drips */
    return sseDripScript(res, 8, 150, '```js\n"AWORK r2"\n```');
  if (last.indexOf('BWORK') >= 0)
    return replyJson(res, '```js\nfinal("B done")\n```');

  return replyJson(res, '```js\nfinal("unexpected path")\n```');
}

function sleep(ms) {
  return new Promise(function (r) { setTimeout(r, ms); });
}

async function main() {
  // Mock server: first free of a few pseudo-random ports (same recipe as
  // rlm_mock_test.js).
  let server = null, port = 0;
  const base = 19000 + (Date.now() % 4000);
  for (let i = 0; i < 6; i++) {
    const p = base + i * 137;
    try {
      server = sofuu.createServer(handler);
      server.listen(p, '127.0.0.1');
      port = p;
      break;
    } catch (e) { server = null; }
  }
  if (!server) { console.log('FAIL could not bind any mock port'); process.exit(1); }
  await sleep(200);

  const drv = {
    provider: 'openai',
    base_url: 'http://127.0.0.1:' + port + '/v1/chat/completions',
    api_key: 'x',
    model: 'mock',
  };

  check('latch starts clear', globalThis.__rlm_aborted !== true);

  // Episode A: long-lived (slow drips, cycles until wall/abort).
  const t0 = Date.now();
  const pA = sofuu.rlm.query('ctx-a', 'LATCH-A probe', Object.assign({}, drv, { maxWallMs: 4000 }))
    .then(function (r) { return { kind: 'resolved', r: r, ms: Date.now() - t0 }; },
          function (e) { return { kind: 'rejected', e: String((e && e.message) || e), ms: Date.now() - t0 }; });
  await sleep(50);

  // Episode B: shorter drips — its watcher tick will kill it first.
  const tB = Date.now();
  const pB = sofuu.rlm.query('ctx-b', 'LATCH-B probe', Object.assign({}, drv, { maxWallMs: 4000 }))
    .then(function (r) { return { kind: 'resolved', r: r, ms: Date.now() - tB }; },
          function (e) { return { kind: 'rejected', e: String((e && e.message) || e), ms: Date.now() - tB }; });

  // The Esc: just after A's watcher tick (~200), so B's tick (~250)
  // consumes it first and B resolves while A is still registered.
  await sleep(160);
  globalThis.__rlm_aborted = true;

  const outB = await pB;
  check('B resolved promptly (' + outB.ms + 'ms)', outB.kind === 'resolved' && outB.ms < 2000);

  // THE INVARIANT: B's teardown ran, A is still registered — the shared
  // latch must STILL be set for A. (Only a finally clears it; A's is the
  // only one left.)
  check('latch survives the first teardown while a sibling is live',
        globalThis.__rlm_aborted === true);

  const outA = await pA;
  check('A received the signal: stopped=aborted (' + outA.ms + 'ms)',
        outA.kind === 'resolved' && outA.r.stopped === 'aborted' && outA.ms < 3000);

  // P3-13 preserved: last episode out re-arms the latch.
  check('latch re-armed after the last episode exits',
        globalThis.__rlm_aborted === false);

  if (server.stop) server.stop();
  if (server.close) server.close();

  console.log(failures === 0
    ? 'RLM-LATCH TEST: ALL PASSED'
    : 'RLM-LATCH TEST: ' + failures + ' FAILURE(S)');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(function (e) {
  console.log('HARNESS ERROR: ' + String((e && e.stack) || e));
  process.exit(1);
});
