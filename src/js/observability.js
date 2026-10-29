/* src/js/observability.js -- sofuu.observability.cli, the `sofuu logs` surface.
 *
 * Shipped JS (OBSERVABILITY-SPEC §4). Rust owns the data: rt/events.rs is the
 * only writer of the JSONL envelope and rt/observability.rs is the only reader
 * of it, so this file never validates - it turns events the host already
 * checked into something a human can scan in a terminal. Rendering lives in JS
 * on purpose (§0): the binary is size-capped at 5,767,168 B, and a column tweak
 * here costs a few hundred bytes of gzipped bytecode instead of a core rebuild.
 *
 * The seam evals this AFTER rt/observability.rs registered runs /
 * lastCheckpoint / postmortem / crashCard, so the namespace is MERGED, never
 * replaced - the same discipline as agent.js and web.js. Only cli, renderRun
 * and VERSION are added here.
 *
 * Everything is best-effort by contract: `sofuu logs` in a project that never
 * ran an agent prints "no runs yet" instead of throwing, and a host half that
 * is missing from this build degrades to reading the two log paths §1 fixes
 * directly. No command in this file can exit non-zero because a log is absent;
 * it exits non-zero only for USAGE errors and unreadable files the operator
 * explicitly named.
 *
 * Chrome (box, table, rule) is ASCII only: the logs get pasted into issues and
 * pulled through `less`/`grep` on terminals whose font support is unknown.
 */
(function () {
  'use strict';
  var g = globalThis;
  if (typeof g.sofuu === 'undefined' || g.sofuu === null) g.sofuu = {};
  var sofuu = g.sofuu;
  /* Never clobber the host namespace: rt/observability.rs put the data
   * functions there before this file eval'd. */
  if (!sofuu.observability || typeof sofuu.observability !== 'object') {
    sofuu.observability = {};
  }
  var OBS = sofuu.observability;
  var VERSION = '0.1.0';

  /* Log locations fixed by §1. They are also the fallback when the Rust
   * lister is unavailable, which is what keeps `show`/`follow` useful while
   * the observability backend is still landing. */
  var LOGS_DIR = '.sofuu/logs';
  var SESSIONS_DIR = '.sofuu/sessions';
  var POLL_MS = 250;          // §4: poll the JSONL tail at 4 Hz
  var IDLE_MS = 120000;       // no growth this long -> degrade to show
  var KIND_W = 16;            // trace kind column
  var STEP_W = 3;             // trace step column
  var CARD_W = 68;            // post-mortem box inner width
  var CARD_LABEL_W = 9;

  /* ── formatting helpers ──────────────────────────────────────────────── */

  function pad(s, n) {
    s = String(s == null ? '' : s);
    if (s.length >= n) return s;
    return s + new Array(n - s.length + 1).join(' ');
  }

  function lpad(s, n) {
    s = String(s == null ? '' : s);
    if (s.length >= n) return s;
    return new Array(n - s.length + 1).join(' ') + s;
  }

  function repeat(c, n) {
    return n > 0 ? new Array(n + 1).join(c) : '';
  }

  /* One line, no control characters, clipped. A payload legitimately carries
   * model text (newlines included); a trace row must stay one row. */
  function clip(s, n) {
    var t = String(s == null ? '' : s)
      .replace(/[\r\n\t]+/g, ' ')
      .replace(/[\u0000-\u001f\u007f]/g, '');
    t = t.replace(/\s+$/, '');
    if (n > 3 && t.length > n) t = t.slice(0, n - 3) + '...';
    return t;
  }

  function one(v) {
    return String(v == null || v === '' ? '?' : v);
  }

  function num(v) {
    if (v === null || v === undefined || v === '') return '-';
    var n = Number(v);
    return isFinite(n) ? String(Math.round(n)) : '-';
  }

  function quoted(v, n) {
    return '"' + clip(v, n) + '"';
  }

  /* ISO 8601 -> "HH:MM:SS.mmm". The date belongs in the run header (a run is
   * one day long at worst); dropping it buys eleven columns of payload. */
  function clock(ts) {
    var s = typeof ts === 'string' ? ts : '';
    if (!s) return '-';
    var t = s.indexOf('T');
    if (t >= 0) s = s.slice(t + 1);
    s = s.replace(/Z$/, '');
    if (s.indexOf('.') < 0) s += '.000';
    return s.slice(0, 12);
  }

  /* runs() reports mtime; seconds (fs metadata) and milliseconds (Date.now)
   * are both in the wild, and the column must agree with the UTC event
   * timestamps it is read next to (§0: ts is ISO/UTC). */
  function stamp(mtime) {
    var n = Number(mtime);
    if (!isFinite(n) || n <= 0) return '-';
    var d = new Date(n > 1e11 ? n : n * 1000);
    var iso;
    try { iso = d.toISOString(); } catch (e) { return '-'; }
    return iso.slice(0, 10) + ' ' + iso.slice(11, 19);
  }

  function wrap(text, width) {
    var words = String(text == null ? '' : text).split(/\s+/);
    var out = [];
    var line = '';
    for (var i = 0; i < words.length; i++) {
      if (!words[i]) continue;
      if (line && line.length + 1 + words[i].length > width) {
        out.push(line);
        line = words[i];
      } else {
        line = line ? line + ' ' + words[i] : words[i];
      }
    }
    if (line) out.push(line);
    if (!out.length) out.push('');
    return out;
  }

  /* ── envelope helpers ────────────────────────────────────────────────── */

  /* The canonical shape keeps fields under `payload`; the drivers have also
   * stamped them inline and events.rs accepts both, so the renderer does too
   * (§0: one validator, both shapes -- never a second format). */
  function payloadOf(ev) {
    if (ev && ev.payload && typeof ev.payload === 'object') return ev.payload;
    return ev || {};
  }

  function stepOf(ev) {
    var p = payloadOf(ev);
    var n = p.step;
    if (n === undefined || n === null || n === '') n = ev && ev.step;
    return n === undefined || n === null || n === '' ? '-' : String(n);
  }

  /* Durations arrive as payload.ms (events.rs tool_result), duration_ms (the
   * agent driver) or durationMs (JS callers): one measurement, three names. */
  function msOf(ev) {
    var p = payloadOf(ev);
    var n = p.ms;
    if (n === undefined || n === null) n = p.duration_ms;
    if (n === undefined || n === null) n = p.durationMs;
    if (n === null || n === undefined || n === '') return 0;
    n = Number(n);
    return isFinite(n) && n > 0 ? Math.round(n) : 0;
  }

  /* Tool argument KEYS only. Command lines and tokens land in tool args, and
   * a trace is a file people paste into bug reports (the same rule the
   * agent-run progress line in main.rs follows). */
  function argKeys(args) {
    var v = args;
    if (typeof v === 'string') {
      try { v = JSON.parse(v); } catch (e) { return ''; }
    }
    if (!v || typeof v !== 'object') return '';
    var out = [];
    for (var k in v) {
      if (Object.prototype.hasOwnProperty.call(v, k)) out.push(k);
    }
    return out.join(', ');
  }

  /* §5 status vocabulary, derived from the last terminal event when the
   * lister did not compute one: run_complete -> OK, halt/fatal_error ->
   * HALTED, retry without a terminal -> RETRYING, else RUNNING. */
  function deriveStatus(entry, events) {
    if (entry && entry.status) return String(entry.status).toUpperCase();
    for (var i = events.length - 1; i >= 0; i--) {
      var k = String((events[i] && events[i].kind) || '');
      if (k === 'run_complete') return 'OK';
      if (k === 'stop') return 'OK';
      if (k === 'halt' || k === 'fatal_error') return 'HALTED';
      if (k === 'retry') return 'RETRYING';
    }
    return events.length ? 'RUNNING' : '-';
  }

  /* Terminal kinds: the agent run's own (run_complete, halt, fatal_error) and
   * the chat session's 'stop' (events.rs STOP_REASONS) - a session log tails
   * forever otherwise, because nothing else marks a finished turn stream. */
  function terminalKind(events) {
    for (var i = events.length - 1; i >= 0; i--) {
      var k = String((events[i] && events[i].kind) || '');
      if (k === 'run_complete' || k === 'halt' || k === 'fatal_error' || k === 'stop') return k;
    }
    return '';
  }

  function lastOfKind(events, kinds) {
    for (var i = events.length - 1; i >= 0; i--) {
      var k = String((events[i] && events[i].kind) || '');
      for (var j = 0; j < kinds.length; j++) {
        if (k === kinds[j]) return events[i];
      }
    }
    return null;
  }

  /* ── host bridge ─────────────────────────────────────────────────────── */

  /* Host functions take ONE JSON string and answer with one JSON string (the
   * sofuu.events.append convention). A missing function, a non-string answer
   * or a throw collapses to null so every caller can degrade instead of
   * dying - the backend is optional by contract. */
  function host(fn, args) {
    try {
      if (!OBS || typeof OBS[fn] !== 'function') return null;
      var raw = OBS[fn](JSON.stringify(args || {}));
      if (typeof raw === 'string') return raw.trim() ? JSON.parse(raw) : null;
      return raw && typeof raw === 'object' ? raw : null;
    } catch (e) {
      return null;
    }
  }

  /* JSONL -> {events, bad}. A torn trailing line is a run cut off mid-write,
   * which is exactly the run somebody is debugging: count it, never throw. */
  function parseLines(text) {
    var rows = String(text == null ? '' : text).split('\n');
    var events = [];
    var bad = 0;
    for (var i = 0; i < rows.length; i++) {
      var s = rows[i].replace(/^\s+|\s+$/g, '');
      if (!s) continue;
      try {
        var v = JSON.parse(s);
        if (v && typeof v === 'object') events.push(v);
        else bad++;
      } catch (e) {
        bad++;
      }
    }
    return { events: events, bad: bad };
  }

  /* Full read: the follow loop needs the byte length to notice growth, and
   * `show` needs the unparseable-line count. */
  async function readLog(path) {
    var out = { text: '', events: [], bad: 0, error: '' };
    try {
      out.text = String(await sofuu.fs.readFile(path, 'utf8'));
    } catch (e) {
      out.error = String((e && e.message) || e);
      return out;
    }
    var parsed = parseLines(out.text);
    out.events = parsed.events;
    out.bad = parsed.bad;
    return out;
  }

  async function fileExists(path) {
    try { return !!(await sofuu.fs.exists(path)); } catch (e) { return false; }
  }

  function runRows() {
    var res = host('runs', { project: '.' });
    if (!res || !Array.isArray(res.runs)) return null;
    return res.runs;
  }

  function findRow(rows, runId) {
    if (!rows) return null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i] && String(rows[i].run_id) === String(runId)) return rows[i];
    }
    return null;
  }

  /* Resolve a run id to its log file: the lister first (it knows about the
   * driver's logDir override), then the two §1 paths by convention. */
  async function resolvePath(runId, row) {
    if (row && row.path) return String(row.path);
    var candidates = [
      LOGS_DIR + '/agent-' + runId + '.jsonl',
      SESSIONS_DIR + '/' + runId + '/events.ndjson',
    ];
    for (var i = 0; i < candidates.length; i++) {
      if (await fileExists(candidates[i])) return candidates[i];
    }
    return null;
  }

  function newestRunId(rows) {
    if (!rows || !rows.length) return '';
    var best = '';
    var seen = -1;
    for (var i = 0; i < rows.length; i++) {
      var n = Number(rows[i] && rows[i].mtime);
      if (!isFinite(n)) n = seen;
      if (n > seen || !best) {
        seen = n;
        best = String(rows[i].run_id);
      }
    }
    return best;
  }

  /* ── trace rendering ─────────────────────────────────────────────────── */

  function summarize(ev, kind) {
    var p = payloadOf(ev);
    var s;
    switch (kind) {
      case 'run_start':
        s = 'agent=' + one(p.agent || p.name);
        if (p.task) s += ' task=' + quoted(p.task, 48);
        return s;
      case 'turn_start':
        return 'turn=' + one(p.turn || p.turn_id || ev.turn_id);
      case 'turn_end':
        return 'turn=' + one(p.turn || p.turn_id) +
          (p.reason ? ' reason=' + one(p.reason) : '');
      case 'tool_call':
      case 'tool':
        return one(p.name || p.tool) + '(' + argKeys(p.args) + ')';
      case 'tool_result':
        /* The duration is the row's trailing column, so it is not repeated
         * here; ok and the error text are what the summary adds. */
        return one(p.name || p.tool) +
          ' ok=' + (p.ok === undefined || p.ok === null ? '?' : String(p.ok === true)) +
          (p.error ? ' error=' + quoted(p.error, 40) : '');
      case 'model_complete':
        return 'in=' + num(p.in === undefined || p.in === null ? p.promptTokens : p.in) +
          ' out=' + num(p.out === undefined || p.out === null ? p.completionTokens : p.out) +
          (p.model ? ' model=' + one(p.model) : '');
      case 'usage':
        return 'in=' + num(p['in']) + ' out=' + num(p.out) +
          ' model=' + one(p.model) + ' provider=' + one(p.provider);
      case 'memory_recall':
        return (p.n === undefined || p.n === null ? '' : 'n=' + num(p.n) + ' ') +
          'query=' + quoted(p.query || p.text, 40);
      case 'memory_store':
        return (p.n === undefined || p.n === null ? '' : 'n=' + num(p.n) + ' ') +
          'text=' + quoted(p.text, 40);
      case 'checkpoint':
        /* The step is already the row's own column; what a resume needs to
         * know about a checkpoint is how much context it carries. */
        return 'messages=' + (Array.isArray(p.messages) ? String(p.messages.length) : num(p.messages)) +
          (p.model ? ' model=' + one(p.model) : '') +
          (p.last_tool ? ' last_tool=' + one(p.last_tool) : '');
      case 'retry':
        return 'attempt=' + num(p.attempt) + ' code=' + one(p.code || p.error_code);
      case 'error':
      case 'fatal_error':
        return 'code=' + one(p.code || p.error_code) + ' ' +
          quoted(p.message || p.error || '', 56);
      case 'halt':
        return 'reason=' + one(p.reason || p.code || p.message);
      case 'run_complete':
        s = 'steps=' + num(p.steps);
        if (p.answer || p.text) s += ' answer=' + quoted(p.answer || p.text, 40);
        return s;
      case 'warning':
      case 'warn':
        return quoted(p.message || p.text || '', 64);
      case 'delta':
      case 'answer_delta':
      case 'stream_chunk':
        return 'text=' + quoted(p.text || p.delta, 48);
      case 'answer':
        return 'answer=' + quoted(p.answer || p.text, 48);
      case 'stop':
        return 'reason=' + one(p.reason);
      case 'served':
        return 'by=' + one(p.by);
      case 'ctx':
        return 'used=' + num(p.used) + ' limit=' + num(p.limit);
      default:
        try { return clip(JSON.stringify(p), 72); } catch (e) { return ''; }
    }
  }

  /* One trace row: clock, kind, step, summary, duration. Fixed columns so two
   * runs diff by eye and greps stay aligned. */
  function renderLine(ev) {
    var kind = ev && ev.kind ? clip(ev.kind, KIND_W - 1) : '?';
    var ms = msOf(ev);
    var line = clock(ev && ev.ts) + '  ' + pad(kind, KIND_W) +
      ' step ' + lpad(stepOf(ev), STEP_W) + '  ' + summarize(ev, kind);
    if (ms) line += '  ' + ms + 'ms';
    return line;
  }

  /* seq is the total order events.rs enforces (§0); sort a COPY so a caller's
   * array is never mutated, and keep input order for events without one. */
  function inSeqOrder(events) {
    var rows = [];
    for (var i = 0; i < events.length; i++) {
      rows.push({ ev: events[i], i: i, seq: Number(events[i] ? events[i].seq : NaN) });
    }
    rows.sort(function (a, b) {
      var x = isFinite(a.seq) ? a.seq : Infinity;
      var y = isFinite(b.seq) ? b.seq : Infinity;
      if (x === y) return a.i - b.i;
      return x < y ? -1 : 1;
    });
    var out = [];
    for (var k = 0; k < rows.length; k++) out.push(rows[k].ev);
    return out;
  }

  /* Render a run as a fixed-width trace. Accepts the parsed envelope array or
   * the raw JSONL text, because the CLI reads files while a host may have
   * already parsed them - a caller cannot get this wrong. Returns the text;
   * printing stays with the caller (and with the console the host wired). */
  function renderRun(events) {
    var list = events;
    if (typeof list === 'string') list = parseLines(list).events;
    if (!Array.isArray(list) || !list.length) return '';
    var ordered = inSeqOrder(list);
    var out = [];
    for (var i = 0; i < ordered.length; i++) out.push(renderLine(ordered[i]));
    return out.join('\n');
  }

  /* ── post-mortem box ─────────────────────────────────────────────────── */

  function cardRow(label, value) {
    var bodyW = CARD_W - CARD_LABEL_W;
    var parts = wrap(value, bodyW);
    var rows = [];
    for (var i = 0; i < parts.length; i++) {
      rows.push('| ' + (i === 0 ? pad(label, CARD_LABEL_W) : pad('', CARD_LABEL_W)) +
        pad(clip(parts[i], bodyW), bodyW) + ' |');
    }
    return rows;
  }

  function cardTitle(text) {
    return '| ' + pad(clip(text, CARD_W), CARD_W) + ' |';
  }

  function cardRule() {
    return '+' + repeat('-', CARD_W + 2) + '+';
  }

  /* The §Component 3 card, rendered. Field labels are ours; every VALUE comes
   * from the card, including error.code (the taxonomy is Rust's to enforce -
   * this function must never invent or translate a code). */
  function renderCard(card) {
    card = card || {};
    var err = (card.error && typeof card.error === 'object') ? card.error : {};
    var ctx = (card.context_snapshot && typeof card.context_snapshot === 'object')
      ? card.context_snapshot : {};
    var rows = [];
    rows.push(cardRule());
    rows.push(cardTitle('CRASH CARD  ' + one(card.crash_id)));
    rows.push(cardRule());
    rows.push.apply(rows, cardRow('run', one(card.run_id)));
    if (card.timestamp) rows.push.apply(rows, cardRow('when', one(card.timestamp)));
    var step = card.failed_step;
    rows.push.apply(rows, cardRow('failed', step === undefined || step === null || step === ''
      ? 'step unknown'
      : 'step ' + String(step)));
    if (err.code || err.message) {
      /* The code is printed verbatim: the taxonomy is Rust's to enforce, so
       * this renderer never maps or invents one. */
      var errText = err.code ? String(err.code) : '';
      if (err.code && err.message) errText += ' ';
      errText += String(err.message || '');
      rows.push.apply(rows, cardRow('error', errText));
    }
    var bits = [];
    if (ctx.model) bits.push('model=' + String(ctx.model));
    if (ctx.provider) bits.push('provider=' + String(ctx.provider));
    if (ctx.context_tokens) bits.push('context_tokens=' + String(ctx.context_tokens));
    if (bits.length) rows.push.apply(rows, cardRow('context', bits.join('  ')));
    if (ctx.last_user_message) {
      rows.push.apply(rows, cardRow('last', 'user: ' + quoted(ctx.last_user_message, 80)));
    }
    if (ctx.last_tool) {
      rows.push.apply(rows, cardRow('last', 'tool: ' + String(ctx.last_tool)));
    }
    /* The hint is the card's own recovery advice. When a card was derived
     * from an event (no postmortem was ever written) there is none, and the
     * only advice this file can honestly give is the resume command. */
    var hint = card.recovery_hint
      ? String(card.recovery_hint)
      : 'sofuu resume ' + String(one(card.run_id)) + ' continues from the last checkpoint';
    rows.push.apply(rows, cardRow('hint', hint));
    if (card.trace_path) rows.push.apply(rows, cardRow('trace', String(card.trace_path)));
    if (err.stack) {
      var stackLines = String(err.stack).split('\n').slice(0, 4);
      for (var i = 0; i < stackLines.length; i++) {
        if (stackLines[i].replace(/\s+/g, '')) {
          rows.push.apply(rows, cardRow(i === 0 ? 'stack' : '', stackLines[i]));
        }
      }
    }
    rows.push(cardRule());
    return rows.join('\n');
  }

  /* When no card was written (crash cards land later than the event), build
   * the same shape from the run's last failure so `--last-error` still
   * answers the only question an operator has: what broke, and what now.
   * The crash_id says (derived) so nobody mistakes it for a real card. */
  function cardFromEvent(ev, runId, path) {
    var p = payloadOf(ev);
    return {
      crash_id: '(derived from the event log)',
      run_id: String(runId || p.run_id || p.sid || '?'),
      timestamp: String((ev && ev.ts) || ''),
      failed_step: p.step,
      error: {
        code: p.code || p.error_code || '',
        message: p.message || p.error || '',
      },
      context_snapshot: {},
      recovery_hint: '',
      trace_path: path || '',
    };
  }

  /* ── commands ────────────────────────────────────────────────────────── */

  function printUsage() {
    console.log('usage: sofuu logs [list]');
    console.log('       sofuu logs show <run_id>      render one run event trace');
    console.log('       sofuu logs --follow [run_id]  stream a run (Ctrl-C exits)');
    console.log('       sofuu logs --last-error       post-mortem box for the newest crash');
  }

  async function cmdList() {
    var rows = runRows();
    if (rows === null) {
      console.log('no runs yet (sofuu.observability.runs is not available in this build)');
      return 0;
    }
    if (!rows.length) {
      console.log('no runs yet (agent runs log to ' + LOGS_DIR + '/agent-<run_id>.jsonl)');
      return 0;
    }
    var idW = 24;
    var startedW = 19;
    console.log(pad('RUN ID', idW) + '  ' + pad('STARTED (UTC)', startedW) +
      '  ' + lpad('STEPS', 5) + '  STATUS');
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i] || {};
      var id = String(r.run_id == null ? '?' : r.run_id);
      var steps = r.steps === undefined || r.steps === null ? '-' : String(r.steps);
      console.log(pad(clip(id, idW), idW) + '  ' + pad(stamp(r.mtime), startedW) +
        '  ' + lpad(steps, 5) + '  ' + deriveStatus(r, []));
    }
    console.log('');
    console.log(rows.length + ' run' + (rows.length === 1 ? '' : 's') +
      '. `sofuu logs show <run_id>` renders one, `sofuu logs --follow` streams it.');
    return 0;
  }

  async function locate(runId) {
    var rows = runRows();
    var row = findRow(rows, runId);
    var path = await resolvePath(runId, row);
    return { row: row, rows: rows, path: path };
  }

  async function cmdShow(rest) {
    var runId = rest[0] ? String(rest[0]) : '';
    if (!runId) {
      console.error('logs: show needs a run id (sofuu logs lists them)');
      return 2;
    }
    var found = await locate(runId);
    if (!found.path) {
      console.log('no run "' + runId + '" yet (looked for ' + LOGS_DIR + '/agent-' +
        runId + '.jsonl and ' + SESSIONS_DIR + '/' + runId + '/events.ndjson)');
      return 1;
    }
    var read = await readLog(found.path);
    if (read.error) {
      console.error('logs: cannot read ' + found.path + ': ' + read.error);
      return 1;
    }
    if (!read.events.length) {
      console.log('no events in ' + found.path + ' yet');
      return 0;
    }
    console.log('run ' + runId + '  status ' + deriveStatus(found.row, read.events) +
      '  ' + read.events.length + ' event' + (read.events.length === 1 ? '' : 's') +
      (read.bad ? '  (' + read.bad + ' unparseable line' + (read.bad === 1 ? '' : 's') + ' skipped)' : ''));
    console.log('path ' + found.path);
    console.log('');
    console.log(renderRun(read.events));
    return 0;
  }

  async function cmdFollow(rest) {
    var runId = rest[0] ? String(rest[0]) : '';
    var found;
    if (!runId) {
      var rows = runRows();
      runId = newestRunId(rows);
      if (!runId) {
        console.log('no runs yet (nothing to follow)');
        return 0;
      }
      found = await locate(runId);
    } else {
      found = await locate(runId);
    }
    if (!found.path) {
      console.log('no run "' + runId + '" yet (looked for ' + LOGS_DIR + '/agent-' +
        runId + '.jsonl and ' + SESSIONS_DIR + '/' + runId + '/events.ndjson)');
      return 1;
    }
    var read = await readLog(found.path);
    if (read.error) {
      console.error('logs: cannot read ' + found.path + ': ' + read.error);
      return 1;
    }
    var printed = read.events.length;
    var terminal = terminalKind(read.events);
    console.log('following ' + runId + ' (' + found.path + ', poll ' + POLL_MS + 'ms, Ctrl-C exits)');
    if (printed) {
      console.log('');
      console.log(renderRun(read.events));
      console.log('');
    } else {
      console.log('no events yet; waiting for the run to write its first line');
    }
    /* Degrade to `show`: nothing more will ever be appended to a run whose
     * log ends in a terminal event, so tailing it would hang forever. */
    if (terminal) {
      console.log('run is ' + deriveStatus(found.row, read.events) +
        '; the log ends in ' + terminal + ', so this trace is complete');
      return 0;
    }
    var lastLen = read.text.length;
    var misses = 0;
    var idle = 0;
    for (;;) {
      await sleep(POLL_MS);
      var text;
      try {
        text = String(await sofuu.fs.readFile(found.path, 'utf8'));
        misses = 0;
      } catch (e) {
        /* Three consecutive misses: the file was rotated away or the bridge
         * is gone. Print what we already printed once and stop - a follow
         * loop that cannot read must not spin forever. */
        misses++;
        if (misses >= 3) {
          console.error('logs: cannot read ' + found.path + ' (' +
            String((e && e.message) || e) + '); stopping the follow');
          return 0;
        }
        continue;
      }
      if (text.length === lastLen) {
        /* Degrade to show: a log that has not grown for IDLE_MS is not being
         * tailed, it is being read. Returning beats hanging in a pipeline;
         * the trace above is already the whole story so far. */
        idle++;
        if (idle * POLL_MS >= IDLE_MS) {
          console.log('no new events in ' + Math.round(IDLE_MS / 1000) +
            's; stopping the follow (re-run --follow to keep streaming)');
          return 0;
        }
        continue;
      }
      idle = 0;
      lastLen = text.length;
      var parsed = parseLines(text);
      if (parsed.events.length < printed) {
        console.log('(log shrank; re-reading from the top)');
        printed = 0;
      }
      for (var i = printed; i < parsed.events.length; i++) {
        console.log(renderLine(parsed.events[i]));
      }
      printed = parsed.events.length;
      var term = terminalKind(parsed.events);
      if (term) {
        console.log('run is ' + deriveStatus(found.row, parsed.events) +
          '; the log ends in ' + term + ', following stops here');
        return 0;
      }
    }
  }

  async function cmdLastError(rest) {
    var runId = rest[0] ? String(rest[0]) : '';
    var args = { project: '.' };
    if (runId) args.run_id = runId;
    var res = host('crashCard', args);
    var card = res && res.ok ? (res.card || null) : null;
    if (card && typeof card === 'object' && (card.crash_id || card.error)) {
      console.log(renderCard(card));
      return 0;
    }
    /* No card on disk. Rather than "nothing to see", answer the question from
     * the newest run whose log ends in a failure - clearly labelled, because
     * it is not a crash card. */
    var rows = runRows() || [];
    var order = runId ? [runId] : [];
    if (!order.length) {
      for (var i = rows.length - 1; i >= 0; i--) order.push(String(rows[i].run_id));
    }
    for (var j = 0; j < order.length; j++) {
      var found = await locate(order[j]);
      if (!found.path) continue;
      var read = await readLog(found.path);
      var ev = lastOfKind(read.events, ['fatal_error', 'halt', 'error']);
      if (!ev) continue;
      console.log(renderCard(cardFromEvent(ev, order[j], found.path)));
      return 0;
    }
    if (!rows.length && !runId) {
      console.log('no crash cards yet and no runs to inspect (' + LOGS_DIR +
        ' and .sofuu/crashes are both empty)');
      return 0;
    }
    console.log('no crash cards yet (a fatal error writes ' +
      '.sofuu/crashes/<crash_id>.json; no run log carries a failure either)');
    return 0;
  }

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  /* argv[0] is the subcommand (main.rs hands over its own args, already split
   * from `sofuu logs`). Returns the process exit code: 0 ok, 1 the named
   * thing is missing or unreadable, 2 a usage error. */
  async function cli(argv) {
    var a = Array.isArray(argv) ? argv.map(String) : [];
    var cmd = a[0] ? a[0] : 'list';
    var rest = a.slice(1);
    try {
      if (cmd === 'list' || cmd === 'ls') return await cmdList();
      if (cmd === 'show') return await cmdShow(rest);
      if (cmd === 'follow') return await cmdFollow(rest);
      if (cmd === 'last-error' || cmd === 'lastError' || cmd === 'last') {
        return await cmdLastError(rest);
      }
      if (cmd === 'help' || cmd === '--help' || cmd === '-h') {
        printUsage();
        return 0;
      }
      console.error('logs: unknown subcommand "' + cmd + '"');
      printUsage();
      return 2;
    } catch (e) {
      /* Best-effort by contract: no path through the renderer may take the
       * process down with a stack trace where a one-line message will do. */
      console.error('logs: ' + String((e && e.message) || e));
      return 1;
    }
  }

  OBS.cli = cli;
  OBS.renderRun = renderRun;
  if (!OBS.VERSION) OBS.VERSION = VERSION;
})();
