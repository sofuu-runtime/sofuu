// tools.js — built-in coding tools for the agent runtime (read/write/edit/
// grep/glob/list_dir/bash over the native sofuu.fs / sofuu.spawn primitives).
//
// Same seam as web.js: pure JS, no host functions, exposes MCP-shaped tool
// specs under sofuu.tools.TOOLS. Agents opt in with tools: ["code"] (all)
// or by individual name; agent.js resolves them like web tools (inline
// tools still win name clashes).
//
// Safety rails (v1):
//   - write_file/edit_file are jailed to the process cwd (lexical resolve,
//     ".." and absolute escapes rejected).
//   - read_file/grep/glob/list_dir are read-only but jailed to cwd the same
//     way (2026-10-15: read_file was the odd one out — any absolute path
//     outside the project reached the provider transcript under every
//     permission profile).
//   - bash runs /bin/sh -c with a hard timeout (default 60s, max 600s) and
//     kills the child on expiry; stdout capped. A shell can still touch
//     anything the user can — set SOFUU_NO_SHELL=1 to strip bash entirely.
//   - All readers refuse credential stores — .sofuu/config.json, .ssh/*,
//     extensionless SSH keys (id_rsa/id_ed25519/id_ecdsa), .aws/credentials,
//     *.pem|.key|.p12|.pfx — and grep/glob skip them entirely (js-5: keys
//     never reach the provider transcript).
//   - Every result is additionally capped by the agent loop's P3
//     tool-result truncation.

(function () {
  if (typeof sofuu === 'undefined' || !sofuu) sofuu = {};
  sofuu.tools = sofuu.tools || {};
  if (sofuu.tools._loaded) return;

  var READ_CAP_CHARS = 60000;
  var GREP_MAX_FILES = 1500;
  var GREP_MAX_MATCHES = 50;
  var GLOB_MAX = 200;
  var BASH_OUT_CAP = 16000;
  var BASH_ERR_CAP = 4096;
  var BASH_DEFAULT_TIMEOUT_MS = 60000;
  /* LONG-HORIZON: real builds and test batteries legitimately run for
   * many minutes — the old 180s cap killed them at the agent's long
   * horizon. 10 minutes covers full cargo/maven/npm pipelines; the
   * caller (or the model via timeout_ms) can still go lower. */
  var BASH_MAX_TIMEOUT_MS = 600000;
  /* Catastrophic-command deny-list (P3). Deliberately SHORT and unambiguous:
   * root/wildcard destruction, disk imagery, fork bombs, host power. Not a
   * sandbox — the approval gate still runs first — this is the second look
   * for the commands that should never run no matter who approves them. */
  var BASH_DENY = [
    /\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)*(-[a-zA-Z]*[rf][a-zA-Z]*\s+)*\/(\s|$)/,       // rm -rf /
    /\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+\/\*/,                                             // rm -r /* (broad wildcard)
    /\brm\s+-[a-zA-Z]*r[a-zA-Z]*[^ ]*\s+\/[a-z]+(\s|$)/,                                // rm -rf /usr, /etc, /tmp …
    /\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(~|\$HOME)(\/|\s|$)/,                           // rm -rf ~ (P3-8: home destruction)
    /\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+\.\.?(\/|\s|$)/,                                // rm -rf .. (parent of cwd)
    /\bgit\s+clean\s+-[a-zA-Z]*x[a-zA-Z]*[fd]/,                                          // git clean -xfd (P3-8: nukes ignored files incl. .env)
    /\bcurl\b[^|;]*\|\s*(ba)?sh\b/,                                                      // curl | sh (P3-8: remote code exec)
    /\bwget\b[^|;]*\|\s*(ba)?sh\b/,                                                      // wget | sh (P3-8)
    /mkfs(\.|\s)/,                                                                       // mkfs.*
    /\bdd\s+[^|;]*of=\/dev\/(disk|rdisk|[sh]d|nvme)/,                                    // dd of=<disk device>
    /:\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;\s*:/,                                             // fork bomb
    /\b(shutdown|reboot|halt|poweroff)\b/,                                               // host power
    /\bchown\s+-R\s+\w+\s+\/(\s|$)/,                                                     // chown -R x /
    /\bchmod\s+-R\s+777\s+\/(\s|$)/,                                                      // chmod -R 777 /
    />\s*\/dev\/(disk|rdisk|[sh]d|nvme)/,                                                // truncate a disk
  ];
  var SKIP_DIRS = { '.git': 1, 'node_modules': 1, 'target': 1, 'dist': 1, '.sofuu': 1 };
  /* S3 (2026-10-23): SOFUU_NO_SHELL is an inventory decision, not just a
   * call-time throw. bash stays out of GROUP (therefore out of every
   * tools:["code"] resolution and every listing a host renders); the
   * execute() rail below still refuses a direct call, defence in depth. */
  var SHELL_DISABLED = false;
  try {
    SHELL_DISABLED = !!(typeof process !== 'undefined' && process.env && process.env.SOFUU_NO_SHELL);
  } catch (eSh) {}

  function cwd() { try { return process.cwd() || '.'; } catch (e) { return '.'; } }

  /* Lexical normalization: collapses '.', '..' and duplicate separators.
   * Relative paths resolve against cwd. Never fails; '..' above the
   * filesystem root just stops climbing. */
  function lexNorm(p) {
    var abs = String(p == null ? '' : p).charAt(0) === '/';
    var parts = (abs ? '' : cwd() + '/') + String(p == null ? '' : p);
    var segs = parts.split('/');
    var out = [];
    for (var i = 0; i < segs.length; i++) {
      var s = segs[i];
      if (!s || s === '.') continue;
      if (s === '..') { if (out.length > 0) out.pop(); }
      else out.push(s);
    }
    return '/' + out.join('/');
  }

  /* S1 (2026-10-23): the canonical root, cached per cwd. On macOS /tmp is
   * /private/tmp, so a canonical target would otherwise print as an
   * absolute path in every result; rel() strips whichever root matches.
   * The cache is keyed by the LEXICAL cwd because a test/host may chdir
   * between runs (the battery does). */
  var ROOT_REAL_CACHE = null; /* { lex, real } */
  async function rootReal() {
    var lex = lexNorm('.');
    if (ROOT_REAL_CACHE && ROOT_REAL_CACHE.lex === lex) return ROOT_REAL_CACHE.real;
    if (!sofuu.fs || typeof sofuu.fs.realpath !== 'function') return null;
    var real = null;
    try { real = await sofuu.fs.realpath('.'); } catch (e) { return null; }
    if (real) ROOT_REAL_CACHE = { lex: lex, real: real };
    return real;
  }

  /* Read-side jail (S1). The four readers used to do a lexical prefix check
   * only, so a symlink planted INSIDE the project (cloned repo, node_modules,
   * or one minted by a prior bash call) passed it and the native read
   * followed the link outside the workspace — the same escape the write
   * jail closed in P1-4. Canonicalise the target (which must exist for a
   * read) and require canonical containment; return the canonical path so
   * the read cannot be redirected by a swapped symlink afterwards. */
  async function readInRoot(p, label) {
    var who = label || 'read';
    var abs = lexNorm(p);
    var root = lexNorm('.');
    if (abs !== root && abs.indexOf(root + '/') !== 0) {
      throw new Error(who + ': path escapes the project directory (reads are jailed to cwd): ' + p);
    }
    var rootRealPath = await rootReal();
    if (rootRealPath) {
      var targetReal = null;
      try { targetReal = await sofuu.fs.realpath(abs); } catch (e) {
        /* Target does not exist (or is unreadable): the lexical check above
         * already ran and the native read will report the real error. */
        return abs;
      }
      if (targetReal !== rootRealPath && targetReal.indexOf(rootRealPath + '/') !== 0) {
        throw new Error(who + ': path resolves (via symlink) outside the project directory: ' + p +
          ' → ' + targetReal);
      }
      return targetReal;
    }
    return abs;
  }

  /* Jail for mutating tools: the resolved path must stay under cwd.
   * P1-4 (AUDIT-2026-09-01): used to be purely LEXICAL — a symlink inside
   * the project pointing outside it (cloned repo, node_modules, or one
   * minted by a prior `ln -s` bash call) passed the prefix check and the
   * write escaped the project. Now symlink-aware: canonicalize the target
   * (sofuu.fs.realpath, native std::fs::canonicalize) and the root, and
   * require the canonical containment. A target that does not exist yet
   * (new file) canonicalizes its NEAREST existing ancestor — a symlink
   * in the leading directories still resolves; a fresh path with no
   * symlinked ancestors falls back to the lexical check, which is exact
   * in that case. */
  async function mustBeInRoot(p) {
    var abs = lexNorm(p);
    var root = lexNorm('.');
    if (abs !== root && abs.indexOf(root + '/') !== 0) {
      throw new Error('path escapes the project directory (writes are jailed to cwd): ' + p);
    }
    if (sofuu.fs && typeof sofuu.fs.realpath === 'function') {
      var rootReal = null;
      try { rootReal = await sofuu.fs.realpath('.'); } catch (eR) {}
      if (rootReal) {
        /* Canonicalize the deepest EXISTING ancestor of the target: walk
         * up until realpath succeeds, then re-attach the missing tail. */
        var probe = abs, tail = '';
        var targetReal = null;
        for (var i = 0; i < 32 && probe && probe !== '/'; i++) {
          try {
            targetReal = await sofuu.fs.realpath(probe);
            break;
          } catch (eP) {
            var slash = probe.lastIndexOf('/');
            if (slash <= 0) break;
            tail = probe.slice(slash + 1) + (tail ? '/' + tail : '');
            probe = probe.slice(0, slash);
          }
        }
        if (targetReal) {
          var canonical = tail ? (targetReal.replace(/\/+$/, '') + '/' + tail) : targetReal;
          if (canonical !== rootReal && canonical.indexOf(rootReal + '/') !== 0) {
            throw new Error('path resolves (via symlink) outside the project directory: ' + p +
              ' → ' + canonical);
          }
          /* P1-12 (AUDIT-2026-09-07): hand the write the CANONICAL path we
           * just vetted, not the lexical one — the native open adds
           * O_NOFOLLOW, so a symlink swapped into the final component after
           * this check now fails with ELOOP instead of being followed, and
           * a symlink that legitimately stays inside the project keeps
           * working because we open its already-vetted real target. */
          return canonical;
        }
        /* targetReal === null: no existing ancestor within reach — the
         * lexical check above already ran, nothing more to verify. */
      }
    }
    return abs;
  }

  function rel(p) {
    var abs = lexNorm(p);
    var root = lexNorm('.');
    if (abs === root) return '.';
    if (abs.indexOf(root + '/') === 0) return abs.slice(root.length + 1);
    /* Canonical results from readInRoot/mustBeInRoot carry the realpath root
     * (e.g. /private/tmp vs /tmp on macOS); strip it too so results stay
     * project-relative. Only when the cache belongs to the current cwd. */
    if (ROOT_REAL_CACHE && ROOT_REAL_CACHE.lex === root) {
      var rr = ROOT_REAL_CACHE.real;
      if (abs === rr) return '.';
      if (abs.indexOf(rr + '/') === 0) return abs.slice(rr.length + 1);
    }
    return abs;
  }

  function parentDir(p) {
    var i = p.lastIndexOf('/');
    return i <= 0 ? '/' : p.slice(0, i);
  }

  async function mkdirp(p) {
    if (!p || p === '/') return;
    try { await sofuu.fs.mkdir(p, { recursive: true }); } catch (e) { /* EEXIST is fine */ }
  }

  /* Directory probe: readdir succeeds only on directories. */
  async function isDir(p) {
    try { await sofuu.fs.readdir(p); return true; } catch (e) { return false; }
  }

  /* Recursive walk with caps. visit(file) is awaited per file. S1: every
   * entry is canonical-checked before it is descended into or read, so a
   * symlink planted INSIDE the project (to /etc, to a home directory, to a
   * sibling repo) cannot smuggle outside content into grep/glob output.
   * Non-symlink entries keep their path — the parent is already canonical,
   * so the join cannot escape. Uses isSymlink when the runtime has it and
   * falls back to canonicalising every entry otherwise. */
  async function walk(root, visit, state) {
    state = state || { files: 0, done: false };
    var canonRoot = await rootReal();
    var queue = [lexNorm(root)];
    var depthGuard = 0;
    while (queue.length && !state.done && depthGuard < 20000) {
      depthGuard++;
      var dir = queue.shift();
      var entries;
      try { entries = await sofuu.fs.readdir(dir); } catch (e) { continue; }
      for (var i = 0; i < entries.length && !state.done; i++) {
        var name = entries[i];
        if (SKIP_DIRS[name]) continue;
        var full = (dir === '/' ? '' : dir) + '/' + name;
        var real = full;
        if (canonRoot) {
          var isLink = true;
          try {
            if (typeof sofuu.fs.isSymlink === 'function') isLink = await sofuu.fs.isSymlink(full);
          } catch (eL) { isLink = true; }
          if (isLink) {
            try { real = await sofuu.fs.realpath(full); } catch (eR) { continue; }
            if (real !== canonRoot && real.indexOf(canonRoot + '/') !== 0) continue;
          }
        }
        if (await isDir(real)) queue.push(real);
        else { state.files++; await visit(real); }
        if (state.files > GREP_MAX_FILES + GLOB_MAX) { state.done = true; }
      }
    }
  }

  function globToRe(g) {
    var re = '';
    g = String(g);
    for (var i = 0; i < g.length; i++) {
      var c = g[i];
      if (c === '*') {
        if (g[i + 1] === '*') { re += '.*'; i++; if (g[i + 1] === '/') i++; }
        else re += '[^/]*';
      } else if (c === '?') re += '[^/]';
      else if ('\\^$.|+()[]{}'.indexOf(c) >= 0) re += '\\' + c;
      else re += c;
    }
    return new RegExp('^' + re + '$');
  }

  function basename(p) { var i = p.lastIndexOf('/'); return i < 0 ? p : p.slice(i + 1); }

  /* Sensitive-file policy, one source of truth (js-5 AUDIT-2026-09-07):
   * returns a reason string when the path is a credential store the agent
   * must not read, null otherwise. read_file throws on it; grep/glob skip
   * the file (they used to walk and print it — both auto-approved — so a
   * routine scan exfiltrated keys into the provider transcript). The user
   * can still paste credentials manually if a task truly needs them.
   * 2026-10-15: extensionless SSH private keys outside .ssh/ and
   * .aws/credentials are unambiguous key material — added; `.env` and
   * similar app config deliberately stay readable (mixed content, often
   * task-necessary). */
  function sensitiveReason(p) {
    var abs = lexNorm(p);
    var base = basename(abs);
    if (/(^|\/)\.sofuu\/config\.json$/.test(abs)) {
      return 'holds API keys — ask the user to provide what you need';
    }
    if (/(^|\/)\.ssh\//.test(abs + '/')) {
      return 'SSH keys are off-limits';
    }
    if (/^(id_rsa|id_ed25519|id_ecdsa|id_dsa)$/.test(base)) {
      return 'SSH private key material — ask the user';
    }
    if (/(^|\/)\.aws\/credentials$/.test(abs)) {
      return 'holds cloud credentials — ask the user';
    }
    if (/\.(pem|key|p12|pfx)$/i.test(base)) {
      return 'private key material — ask the user';
    }
    return null;
  }

  function assertNotSensitive(p) {
    var why = sensitiveReason(p);
    if (why) throw new Error('refusing to read ' + p + ' (' + why + ')');
  }

  var TOOLS = {};

  TOOLS.read_file = {
    name: 'read_file',
    description: 'Read a text file (jailed to the project directory, like the other readers). Returns cat -n style numbered lines so exact text can be quoted into edit_file. For large files pass start/end (1-based, inclusive).',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path (relative to cwd or absolute)' },
        start: { type: 'number', description: 'First line to return (1-based, default 1)' },
        end: { type: 'number', description: 'Last line to return (inclusive)' },
      },
      required: ['path'],
    },
    execute: async function (a) {
      /* Jailed to the project directory like grep/glob/list_dir (2026-10-15),
       * and symlink-checked like the writers (S1, 2026-10-23): an absolute
       * path outside cwd — /etc/passwd, a sibling repo's config — used to
       * stream straight into the provider transcript under every permission
       * profile, and an in-project symlink did the same after the lexical
       * guard landed. */
      var p = await readInRoot(a && a.path, 'read_file');
      assertNotSensitive(p);
      var text = await sofuu.fs.readFile(p, 'utf8');
      var lines = text.split('\n');
      if (lines.length && lines[lines.length - 1] === '') lines.pop();
      var s = Math.max(1, ((a && a.start) | 0) || 1);
      var e = Math.min(lines.length, ((a && a.end) | 0) || lines.length);
      var out = [];
      var total = 0;
      for (var i = s; i <= e; i++) {
        var row = ('     ' + i).slice(-5) + '\t' + lines[i - 1];
        if (total + row.length > READ_CAP_CHARS) {
          out.push('…(' + (e - i + 1) + ' more lines below the ' + READ_CAP_CHARS + ' char cap — re-call with start=' + i + ')');
          break;
        }
        out.push(row);
        total += row.length;
      }
      return rel(p) + ' (' + lines.length + ' lines)' + (out.length ? '\n' + out.join('\n') : ' (empty file)');
    },
  };

  /* ── TUI edit/write diff (2026-10-03) ──────────────────────────
   * edit_file/write_file stash a bounded unified diff here for agent.js
   * to forward on the tool_result EVENT. The execute() return strings
   * below are UNCHANGED (the model reads them; existing tests parse
   * them) — this is display-only data for the TUI driver, whose event
   * results are clipped to 200 chars and could otherwise never show WHAT
   * changed. Single-slot: agent.js consumes + clears it per call, so a
   * stale diff can never attach to the wrong event. */
  function setLastDiff(name, diff) {
    try { globalThis.__sofuu_last_diff = { name: name, diff: String(diff || '') }; }
    catch (e) {}
  }
  /* Bounded unified diff of one replacement. first = byte offset of the
   * (first) match in text. Context lines come from the ORIGINAL text;
   * removed/added lines from oldS/newS. Capped so a huge replacement
   * cannot flood the TUI (or the event payload): callers pass the limits.
   *
   * Row shape is `NNNN mark content` (4-wide right-aligned gutter, mark
   * in {'-','+',' '}, one space, the line): the TUI driver paints the
   * gutter dim and tints +/- rows full-width like a difftool. The payload
   * itself stays ANSI-free — it also travels to the model. Gutter numbers
   * are old-file lines for context/removed rows and new-file lines for
   * added rows (the @@ header carries both starts). */
  var DIFF_MAX_BODY_LINES = 40;
  function diffNum(n) {
    var s = String(n);
    while (s.length < 4) s = ' ' + s;
    return s;
  }
  function buildEditDiff(rel, text, first, oldS, newS, totalCount) {
    function lineOf(off) {
      var n = 0;
      for (var i = 0; i < off; i++) if (text[i] === '\n') n++;
      return n; /* 0-based */
    }
    var startLine0 = lineOf(first);
    /* Lines spanned by the match, expanded to whole lines: a mid-line
     * replacement of "alpha" shows "-one alpha two" / "+one beta two",
     * not the bare "-alpha" / "+beta" the raw strings would give. */
    var endLine0 = lineOf(first + String(oldS).length - 1);
    if (endLine0 < startLine0) endLine0 = startLine0;
    var allOld = text.split('\n');
    var lineStartOff = first;
    while (lineStartOff > 0 && text[lineStartOff - 1] !== '\n') lineStartOff--;
    var oldBlockText = allOld.slice(startLine0, endLine0 + 1).join('\n');
    var newBlockText = oldBlockText.slice(0, first - lineStartOff) + String(newS) +
      oldBlockText.slice(first - lineStartOff + String(oldS).length);
    var oldBlock = oldBlockText.split('\n');
    var newBlock = newBlockText.split('\n');
    var startLine = startLine0 + 1; /* 1-based for the header */
    var CTX = 3;
    var from = Math.max(0, startLine0 - CTX);
    var to = Math.min(allOld.length, startLine0 + oldBlock.length + CTX);
    var merged = [];
    var oldLn = from + 1, newLn = from + 1;
    for (var l = from; l < to; l++) {
      if (l === startLine0) {
        /* The replaced block: old lines out, new lines in. */
        for (var o = 0; o < oldBlock.length; o++) {
          merged.push({ ln: oldLn++, mark: '-', text: oldBlock[o] });
        }
        for (var nI = 0; nI < newBlock.length; nI++) {
          merged.push({ ln: newLn++, mark: '+', text: newBlock[nI] });
        }
        l += oldBlock.length - 1; /* skip the replaced originals */
        continue;
      }
      merged.push({ ln: oldLn, mark: ' ', text: allOld[l] });
      oldLn++; newLn++;
    }
    var rows = [];
    for (var r = 0; r < merged.length && rows.length < DIFF_MAX_BODY_LINES; r++) {
      rows.push(diffNum(merged[r].ln) + ' ' + merged[r].mark + ' ' + merged[r].text);
    }
    var truncated = merged.length > rows.length;
    var out = ['--- a/' + rel + ' (line ' + startLine + ')',
               '+++ b/' + rel,
               '@@ -' + (from + 1) + ',' + (to - from) +
               ' +' + (from + 1) + ',' + (to - from + (newBlock.length - oldBlock.length)) + ' @@']
      .concat(rows);
    if (truncated) out.push('... (diff truncated — file has the rest)');
    if (totalCount > 1) out.push('... (' + totalCount + ' total replacements; showing the first)');
    return out.join('\n');
  }
  /* Bounded preview of written content: every line is new, so there is no
   * old side. Capped — the model already holds the full content (it sent
   * it); duplicating a whole file into the result would bill context for
   * zero information. */
  var WRITE_PREVIEW_LINES = 25;
  function buildWritePreview(rel, content) {
    var lines = String(content).split('\n');
    var out = ['+++ b/' + rel + ' (' + lines.length + ' lines)'];
    var shown = Math.min(lines.length, WRITE_PREVIEW_LINES);
    for (var i = 0; i < shown; i++) out.push(diffNum(i + 1) + ' + ' + lines[i]);
    if (lines.length > shown) out.push('... (' + (lines.length - shown) + ' more lines)');
    return out.join('\n');
  }

  /* P3.1 edit anchors (2026-10-16): write_file/edit_file RESULTS carry a
   * few numbered lines of the post-change file — mirroring read_file's
   * cat -n format (5-wide number, tab), ≤120 chars/line, ≤5 lines — so
   * the model sees the ACTUAL on-disk result around the change instead
   * of trusting its own splice arithmetic (the catch: a "successful"
   * edit result string proves nothing about line numbers or neighbors,
   * which is exactly what the next edit needs). Freshly re-read from
   * disk, best-effort: any failure keeps the plain result string. */
  function anchorRider(text, midIdx, sideLines) {
    try {
      var lines = String(text).split('\n');
      if (midIdx > text.length) midIdx = text.length;
      var mid = text.slice(0, midIdx).split('\n').length - 1; /* 0-based */
      var lo = Math.max(0, mid - sideLines);
      var hi = Math.min(lines.length, mid + sideLines + 1);
      var out = [];
      for (var i = lo; i < hi; i++) {
        var row = lines[i];
        if (row.length > 120) row = row.slice(0, 119) + '…';
        out.push(('     ' + (i + 1)).slice(-5) + '\t' + row);
      }
      return out.length ? '\npost-edit context:\n' + out.join('\n') : '';
    } catch (e) {
      return '';
    }
  }

  TOOLS.write_file = {
    name: 'write_file',
    description: 'Create or overwrite a file with the full given content (jailed to the project directory; parent dirs are created). For targeted changes prefer edit_file.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path relative to cwd' },
        content: { type: 'string', description: 'The complete file content' },
      },
      required: ['path', 'content'],
    },
    execute: async function (a) {
      if (!a || typeof a.content !== 'string') throw new Error('write_file: content must be a string');
      var p = await mustBeInRoot(a.path);
      await mkdirp(parentDir(p));
      var existed = await sofuu.fs.exists(p);
      await sofuu.fs.writeFile(p, a.content);
      setLastDiff('write_file', buildWritePreview(rel(p), a.content));
      var res = (existed ? 'overwrote ' : 'created ') + rel(p) + ' (' + a.content.length + ' chars)';
      try {
        res += anchorRider(await sofuu.fs.readFile(p, 'utf8'), 0, 2);
      } catch (eW) {}
      return res;
    },
  };

  /* -- Phase 5: resilient edits (whitespace-tolerant match + syntax dry-run) --
   * Two failure modes used to cost a whole round-trip each: the model
   * restates a block with indentation the file does not use (the literal
   * match misses), or it emits code that does not parse (the garbage lands
   * on disk and the NEXT step discovers it). Both are cheap to catch here.
   *
   * 1. Whitespace-tolerant fallback. When the literal search finds nothing,
   *    lines are compared with leading/trailing whitespace ignored, and the
   *    match must still be UNIQUE - guessing which of two identical blocks
   *    to rewrite is how edits corrupt files. The new block inherits the
   *    matched block's indentation; relative indentation inside it is kept.
   *
   * 2. Syntax dry-run. For .js/.mjs/.cjs/.ts/.json the candidate is PARSED
   *    before it reaches disk (sofuu.parse compiles, never executes). An
   *    edit that would break a file that currently parses is rejected and
   *    nothing is written. A file that is ALREADY broken stays editable -
   *    fixing it is usually why the agent is editing - and the remaining
   *    error comes back as a warning on the result instead.
   *
   * Both paths are additive: literal matching, replace_all and every result
   * string keep their exact old shape, and the gated extension list is
   * deliberately short (md/txt/py/... edits are never blocked). */

  var SYNTAX_GATED = { js: 1, mjs: 1, cjs: 1, ts: 1, json: 1 };

  function extOf(p) {
    var m = /\.([A-Za-z0-9]+)$/.exec(String(p == null ? '' : p));
    return m ? m[1].toLowerCase() : '';
  }

  function oneLine(s) {
    return String(s == null ? '' : s).replace(/\s+/g, ' ').trim().slice(0, 240);
  }

  /* Verdict for one complete candidate body: { ok, error, line } or null.
   * null means the gate does not apply here (free-form extension, or a host
   * runtime without sofuu.parse) - it NEVER means "clean". */
  function syntaxVerdict(text, p) {
    var ext = extOf(p);
    if (!Object.prototype.hasOwnProperty.call(SYNTAX_GATED, ext)) return null;
    if (ext === 'json') {
      try {
        JSON.parse(text);
        return { ok: true };
      } catch (e) {
        var msg = String((e && e.message) || e);
        var lm = /line (\d+)/i.exec(msg);
        return { ok: false, error: 'JSON.parse: ' + oneLine(msg), line: lm ? (parseInt(lm[1], 10) | 0) : 0 };
      }
    }
    var parse = (typeof sofuu !== 'undefined' && sofuu) ? sofuu.parse : null;
    if (typeof parse !== 'function') return null;
    var isTs = ext === 'ts';
    var asModule = ext === 'mjs' || (isTs && /(^|\n)[ \t]*(import|export)\b/.test(text));
    var r = null;
    try { r = parse(text, { filename: rel(p), module: asModule, ts: isTs }); }
    catch (eP) { return null; }
    if (!r || typeof r.ok !== 'boolean') return null;
    if (r.ok) return { ok: true };
    if (!asModule) {
      /* Classic-script mode rejects legal ESM, so retry as a module before
       * calling the file broken. A module that parses but cannot resolve an
       * import reports a non-SyntaxError type - that is a dependency
       * problem, not a syntax one: no verdict (null), never a rejection. */
      var r2 = null;
      try { r2 = parse(text, { filename: rel(p), module: true, ts: isTs }); } catch (eP2) { r2 = null; }
      if (r2 && r2.ok === true) return { ok: true };
      if (r2 && r2.ok === false && r2.type && r2.type !== 'SyntaxError') return null;
    }
    return { ok: false, line: (r.line | 0), error: oneLine(r.error || 'syntax error') };
  }

  /* Gate a candidate before it is written. Returns '' when the edit may
   * proceed (no verdict, or clean); throws when it would introduce a syntax
   * error into a file that parses today; returns a warning rider when the
   * file was already broken and still is. */
  function syntaxGate(before, after, p) {
    var cand = syntaxVerdict(after, p);
    if (!cand || cand.ok) return '';
    var orig = syntaxVerdict(before, p);
    var where = rel(p) + (cand.line > 0 ? ':' + cand.line : '');
    if (orig && orig.ok === false) {
      return '\n[warning] ' + where + ' still does not parse (' + cand.error +
        ') - the file was already broken before this edit.';
    }
    if (!orig) return '';
    throw new Error('edit_file: syntax dry-run rejected the edit in ' + where + ': ' +
      cand.error + ' - nothing was written to disk; fix the syntax and retry.');
  }

  function normLine(s) {
    return String(s).replace(/[ \t\r]+$/, '').replace(/^[ \t]+/, '');
  }

  /* Line-block matcher that ignores indentation. Returns one entry per
   * candidate region: { start, end, line, indent }, where [start, end) is
   * the byte span of the matched lines WITHOUT the final newline. */
  function tolerantMatches(text, oldS) {
    var fileLines = String(text).split('\n');
    var starts = [], off = 0, i;
    for (i = 0; i < fileLines.length; i++) { starts.push(off); off += fileLines[i].length + 1; }
    var want = String(oldS == null ? '' : oldS).replace(/\r\n/g, '\n').split('\n');
    while (want.length && !want[0].trim()) want.shift();
    while (want.length && !want[want.length - 1].trim()) want.pop();
    if (!want.length) return [];
    var wantN = [];
    for (i = 0; i < want.length; i++) wantN.push(normLine(want[i]));
    var hits = [];
    for (var s = 0; s + wantN.length <= fileLines.length; s++) {
      var ok = true;
      for (var k = 0; k < wantN.length; k++) {
        if (normLine(fileLines[s + k]) !== wantN[k]) { ok = false; break; }
      }
      if (!ok) continue;
      var last = s + wantN.length - 1;
      var lead = /^[ \t]*/.exec(fileLines[s]);
      var tail = fileLines[last].replace(/[ \t\r]+$/, '');
      hits.push({ start: starts[s], end: starts[last] + tail.length,
                  line: s + 1, indent: lead ? lead[0] : '' });
    }
    return hits;
  }

  /* Re-base a candidate block onto the file's own indentation. base is the
   * candidate's first-line indent (where the model thought the block sat),
   * want is the matched block's real indent. A line that does not start with
   * base keeps its own prefix, so deliberate outdents survive. */
  function reindent(block, base, want) {
    var ls = String(block == null ? '' : block).split('\n');
    if (base === want) return ls.join('\n');
    for (var i = 0; i < ls.length; i++) {
      if (!ls[i].trim()) { ls[i] = ''; continue; }
      ls[i] = want + (ls[i].indexOf(base) === 0 ? ls[i].slice(base.length) : ls[i]);
    }
    return ls.join('\n');
  }

  TOOLS.edit_file = {
    name: 'edit_file',
    description: 'Replace a string in a file. old_string must match exactly ONE place (quote it from read_file); add surrounding context if it is ambiguous, or set replace_all. If the text is not found, a whitespace-tolerant line match is tried (it too must be unique) and the block is re-indented to the file. .js/.mjs/.cjs/.ts/.json edits are syntax-checked before writing: an edit that would break a file that parses today is rejected and nothing is written. Jailed to the project directory.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path relative to cwd' },
        old_string: { type: 'string', description: 'The exact text to replace (must be unique unless replace_all)' },
        new_string: { type: 'string', description: 'The replacement text' },
        replace_all: { type: 'boolean', description: 'Replace every occurrence instead of requiring a unique match' },
      },
      required: ['path', 'old_string', 'new_string'],
    },
    execute: async function (a) {
      var p = await mustBeInRoot(a && a.path);
      var oldS = String((a && a.old_string) == null ? '' : a.old_string);
      if (!oldS) throw new Error('edit_file: old_string is required (quote it exactly from read_file)');
      var newS = String((a && a.new_string) == null ? '' : a.new_string);
      var replaceAll = !!(a && a.replace_all);
      var text = await sofuu.fs.readFile(p, 'utf8');
      var count = 0, idx = 0, first = -1;
      while ((idx = text.indexOf(oldS, idx)) >= 0) { count++; if (first < 0) first = idx; idx += oldS.length; }
      if (count > 1 && !replaceAll) {
        throw new Error('edit_file: old_string matches ' + count + ' places in ' + rel(p) +
          ' - include more surrounding context to make it unique, or pass replace_all:true');
      }
      /* removed is what the diff shows as the OLD side: oldS on the literal
       * path, the file's real bytes on the tolerant path (they differ there
       * by indentation - a diff built from oldS would show lines that are
       * not in the file). */
      var removed = oldS;
      var nRep = 1;
      var how = '';
      /* applied is what the file actually receives (the tolerant path
       * re-indents the model's block); the diff must show THESE bytes, not
       * the raw candidate, or the TUI paints lines the file never had. */
      var applied = newS;
      var updated;
      if (count > 0) {
      /* js-6 (AUDIT-2026-09-07): splice literally — String.replace (the old
       * single-match branch) interprets $&, $$, $` and $' in new_string as
       * replacement patterns, silently corrupting the file when the new
       * text contains those two-char sequences. split/join (replace_all)
       * was already literal; the single-match branch now is too. */
        updated = replaceAll ? text.split(oldS).join(newS)
                             : text.slice(0, first) + newS + text.slice(first + oldS.length);
        nRep = replaceAll ? count : 1;
      } else {
        var hits = tolerantMatches(text, oldS);
        if (!hits.length) {
          throw new Error('edit_file: old_string not found in ' + rel(p) +
            ' - read_file it and copy the exact text (no line block matched either, ignoring indentation)');
        }
        if (hits.length > 1 && !replaceAll) {
          throw new Error('edit_file: old_string matches ' + hits.length + ' places in ' + rel(p) +
            ' ignoring indentation (lines ' + hits.map(function (h) { return h.line; }).join(', ') +
            ') - include more surrounding context to make it unique, or pass replace_all:true');
        }
        var base = (/^[ \t]*/.exec(newS) || [''])[0];
        var crlf = text.indexOf('\r\n') >= 0;
        var spans = hits.slice().sort(function (x, y) { return y.start - x.start; });
        updated = text;
        for (var si = 0; si < spans.length; si++) {
          var sp = spans[si];
          var repl = reindent(newS, base, sp.indent);
          /* The span stops before its line terminator, so the file keeps
           * its own break after the block: a replacement that brings a
           * trailing newline would leave a blank line behind, so drop it. */
          while (repl.length && (repl.charAt(repl.length - 1) === '\n' ||
                                 repl.charAt(repl.length - 1) === '\r')) {
            repl = repl.slice(0, repl.length - 1);
          }
          if (crlf && repl.indexOf('\r') < 0) repl = repl.split('\n').join('\r\n');
          updated = updated.slice(0, sp.start) + repl + updated.slice(sp.end);
        }
        first = hits[0].start;
        removed = text.slice(hits[0].start, hits[0].end).replace(/\r/g, '');
        nRep = hits.length;
        how = ' [whitespace-tolerant match]';
        applied = reindent(newS, base, hits[0].indent);
      }
      /* Dry run BEFORE the write: a rejected edit leaves the file
       * byte-identical on disk. */
      var gateWarn = syntaxGate(text, updated, p);
      await sofuu.fs.writeFile(p, updated);
      setLastDiff('edit_file', buildEditDiff(rel(p), text, first, removed, applied, nRep));
      var res = 'edited ' + rel(p) + ': ' + (nRep === 1 ? '1 replacement' : nRep + ' replacements') +
        ' (' + (updated.length - text.length >= 0 ? '+' : '') + (updated.length - text.length) + ' chars)' + how;
      try {
        /* Anchor against the FRESH on-disk bytes, not the in-memory splice
         * - the rider's whole point is independent proof of what happened. */
        res += anchorRider(await sofuu.fs.readFile(p, 'utf8'), Math.min(first, updated.length), 2);
      } catch (eE) {}
      return res + gateWarn;
    },
  };
  TOOLS.grep = {
    name: 'grep',
    description: 'Search file contents with a regular expression across the project (skips .git/node_modules/target/dist). Returns path:line: text matches, capped.',
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'JavaScript regular expression source (e.g. "TODO|FIXME")' },
        path: { type: 'string', description: 'Directory or file to search (default: cwd)' },
        glob: { type: 'string', description: 'Only search files whose name matches this glob (e.g. "*.js")' },
      },
      required: ['pattern'],
    },
    execute: async function (a) {
      var re;
      try { re = new RegExp(a && a.pattern); } catch (e) { throw new Error('grep: invalid pattern: ' + String(e.message || e)); }
      var root = await readInRoot((a && a.path) || '.', 'grep');
      var gfilter = a && a.glob ? globToRe(a.glob) : null;
      var state = { files: 0, done: false };
      var hits = [];
      await walk(root, async function (file) {
        if (state.done) return;
        if (sensitiveReason(file)) return; /* js-5: credential stores never enter the transcript */
        if (gfilter && !gfilter.test(basename(file))) return;
        var text;
        try { text = await sofuu.fs.readFile(file, 'utf8'); } catch (e) { return; }
        if (text.indexOf('\u0000') >= 0) return; /* binary */
        var ls = text.split('\n');
        for (var i = 0; i < ls.length; i++) {
          if (re.test(ls[i])) {
            hits.push(rel(file) + ':' + (i + 1) + ': ' + ls[i].trim().slice(0, 200));
            if (hits.length >= GREP_MAX_MATCHES) { state.done = true; return; }
          }
        }
      }, state);
      if (!hits.length) return 'no matches for /' + (a && a.pattern) + '/';
      return hits.join('\n') + (state.done ? '\n…(capped at ' + GREP_MAX_MATCHES + ' matches)' : '');
    },
  };

  TOOLS.glob = {
    name: 'glob',
    description: 'Find files by glob pattern across the project (** matches across directories, * within one). Returns paths, capped at ' + GLOB_MAX + '.',
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'Glob pattern, e.g. "src/**/*.rs" or "*.md"' },
        path: { type: 'string', description: 'Directory to search (default: cwd)' },
      },
      required: ['pattern'],
    },
    execute: async function (a) {
      var re;
      try { re = globToRe((a && a.pattern) || ''); } catch (e) { throw new Error('glob: invalid pattern'); }
      var root = await readInRoot((a && a.path) || '.', 'glob');
      var state = { files: 0, done: false };
      var found = [];
      await walk(root, async function (file) {
        if (found.length >= GLOB_MAX) { state.done = true; return; }
        if (sensitiveReason(file)) return; /* js-5: credential paths never enter the transcript */
        if (re.test(rel(file)) || re.test(basename(file))) found.push(rel(file));
      }, state);
      if (!found.length) return 'no files match ' + (a && a.pattern);
      return found.join('\n') + (found.length >= GLOB_MAX ? '\n…(capped at ' + GLOB_MAX + ')' : '');
    },
  };

  TOOLS.list_dir = {
    name: 'list_dir',
    description: 'List one directory (non-recursive), directories marked with a trailing "/".',
    parameters: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Directory path (default: cwd)' } },
      required: [],
    },
    execute: async function (a) {
      var p = await readInRoot((a && a.path) || '.', 'list_dir');
      var entries = await sofuu.fs.readdir(p);
      var out = [];
      for (var i = 0; i < entries.length && out.length < 500; i++) {
        var full = (p === '/' ? '' : p) + '/' + entries[i];
        out.push((await isDir(full) ? 'd ' : '  ') + entries[i]);
      }
      (out.sort(function (x, y) { return x.slice(2).localeCompare(y.slice(2)); }));
      return rel(p) + '/ (' + entries.length + ' entries)\n' + out.join('\n');
    },
  };

  TOOLS.bash = {
    name: 'bash',
    /* selfTimed: the agent's outer withTimeout must NOT wrap bash —
     * bash polices its OWN timeout below (per-call timeout_ms, up to
     * 10 minutes); an outer 30s default would strangle every long
     * build the agent runs. */
    selfTimed: true,
    description: 'Run a shell command (/bin/sh -c) in the project directory and return stdout+stderr+exit code. Killed at the timeout (default 60s, max 600s). Prefer specific tools for reading/editing files.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to run' },
        timeout_ms: { type: 'number', description: 'Kill after this many ms (1000-600000, default 60000)' },
      },
      required: ['command'],
    },
    execute: async function (a) {
      if (process.env.SOFUU_NO_SHELL) throw new Error('bash: disabled (SOFUU_NO_SHELL is set)');
      var cmd = String((a && a.command) || '');
      if (!cmd.trim()) throw new Error('bash: command required');
      /* P3 deny-list: a small set of unambiguously catastrophic patterns is
       * refused EVEN when the user approves — a mistyped approval on `rm -rf /`
       * should not need a post-mortem. Matched against the raw command; the
       * tool is not a security boundary (the permission gate is), this only
       * catches the classics before they run. */
      for (var d = 0; d < BASH_DENY.length; d++) {
        if (BASH_DENY[d].test(cmd)) {
          throw new Error('bash: refused — the command matches a catastrophic ' +
            'pattern (deny-list ' + d + '). If you truly need this, run it yourself in a terminal.');
        }
      }
      var timeout = Math.min(BASH_MAX_TIMEOUT_MS, Math.max(1000, ((a && a.timeout_ms) | 0) || BASH_DEFAULT_TIMEOUT_MS));
      return await new Promise(function (resolve) {
        var out = '', errOut = '', done = false, code = -1, tid = null;
        function capAdd(s) { if (out.length < BASH_OUT_CAP) out += s.slice(0, BASH_OUT_CAP - out.length); }
        function capErr(s) { if (errOut.length < BASH_ERR_CAP) errOut += s.slice(0, BASH_ERR_CAP - errOut.length); }
        function finish(tag) {
          if (done) return;
          done = true;
          if (tid !== null) clearTimeout(tid);
          var s = '$ ' + cmd + '\n' + (tag || '');
          if (out) s += out;
          if (errOut) s += '\n[stderr]\n' + errOut;
          if (!tag) s += '\n[exit ' + code + ']';
          resolve(s);
        }
        var shArgs = ['-c', cmd];
        var shCmd = '/bin/sh';
        try {
          if (typeof sofuu !== 'undefined' && sofuu.fs && !sofuu.fs.exists('/bin/sh') && sofuu.fs.exists('/system/bin/sh')) {
            shCmd = '/system/bin/sh';
          }
        } catch (_) {}
        var proc;
        try {
          proc = sofuu.spawn({
            command: shCmd,
            args: shArgs,
            onStdout: function (chunk) { capAdd(String(chunk)); },
            onStderr: function (chunk) { capErr(String(chunk)); },
            onExit: function (c) { code = c; finish(''); },
          });
        } catch (e) {
          resolve('$ ' + cmd + '\nbash: spawn failed: ' + String(e.message || e));
          return;
        }
        tid = setTimeout(function () {
          try { proc.kill(); } catch (e) {}
          finish('\n[killed: timed out after ' + timeout + 'ms]\n');
        }, timeout);
      });
    },
  };

  /* code_eval — In-process deterministic JavaScript/TypeScript sandboxed execution.
   * Powered by sofuu.sandbox. Has zero-subprocess overhead (< 1ms startup),
   * strict instruction fuel limit (500k ops), 2000ms wall-clock deadline,
   * 16MB memory ceiling, captured console logs, and hermetic host isolation. */
  TOOLS.code_eval = {
    name: 'code_eval',
    description: 'Execute JavaScript or TypeScript code in a secure, deterministic in-process sandbox. Use this for math, data manipulation, algorithm verification, parsing, or testing logic. Code runs in < 1ms with captured console.log output. Host OS/filesystem/network APIs are completely disabled for safety.',
    parameters: {
      type: 'object',
      properties: {
        code: {
          type: 'string',
          description: 'The JavaScript/TypeScript code or expression to evaluate. The result of the last expression or return statement is returned.',
        },
        timeout_ms: {
          type: 'number',
          description: 'Optional wall-clock timeout in milliseconds (default 2000, max 10000).',
        },
        fuel_limit: {
          type: 'number',
          description: 'Optional instruction fuel limit (default 500000, max 5000000). Traps infinite loops deterministically.',
        },
      },
      required: ['code'],
    },
    execute: async function (args) {
      var code = (args && typeof args.code === 'string') ? args.code : '';
      if (!code.trim()) throw new Error('code_eval: code string required');
      var timeout = 2000;
      if (args && typeof args.timeout_ms === 'number' && args.timeout_ms > 0) {
        timeout = Math.min(10000, Math.floor(args.timeout_ms));
      }
      var fuel = 500000;
      if (args && typeof args.fuel_limit === 'number' && args.fuel_limit > 0) {
        fuel = Math.min(5000000, Math.floor(args.fuel_limit));
      }

      if (typeof sofuu === 'undefined' || !sofuu.sandbox || typeof sofuu.sandbox.run !== 'function') {
        throw new Error('code_eval: sofuu.sandbox runtime substrate not available');
      }

      var res = sofuu.sandbox.run(code, {
        fuel: fuel,
        timeoutMs: timeout,
        maxMemoryMB: 16,
        captureConsole: true,
        throwOnError: false,
      });

      var out = [];
      if (res.logs && res.logs.length > 0) {
        out.push('=== Console Output ===\n' + res.logs.join('\n'));
      }

      if (res.success) {
        var rStr = '';
        if (typeof res.result === 'undefined') {
          rStr = '(undefined)';
        } else if (typeof res.result === 'string') {
          rStr = res.result;
        } else {
          try {
            rStr = JSON.stringify(res.result, null, 2);
          } catch (eJ) {
            rStr = String(res.result);
          }
        }
        out.push('=== Result ===\n' + rStr);
        out.push('(fuel: ' + (res.fuelUsed || 0) + ' ops, ' + (res.elapsedMs ? res.elapsedMs.toFixed(2) : 0) + 'ms)');
        return out.join('\n\n');
      } else {
        var errType = res.errorType || (res.interrupted ? 'interrupted' : 'error');
        var errDesc = res.error || 'Evaluation failed';
        out.push('=== Execution Failed [' + errType + '] ===\n' + errDesc);
        out.push('(fuel consumed: ' + (res.fuelUsed || 0) + ' ops, ' + (res.elapsedMs ? res.elapsedMs.toFixed(2) : 0) + 'ms)');
        return out.join('\n\n');
      }
    },
  };

  sofuu.tools._loaded = true;
  sofuu.tools.VERSION = '1';
  sofuu.tools.TOOLS = TOOLS;
  /* todo_write — a living checklist the model maintains across a multi-step
   * task (P2, claude-code parity). The steps ride the chat's event stream
   * so hosts render progress; state is one JSON string on a global. */
  TOOLS.todo_write = {
    name: 'todo_write',
    description: 'Maintain the task checklist. Pass the FULL list every time (created/updated in order, status: todo | doing | done). Rewrite it whenever the plan or progress changes — the user watches it live.',
    parameters: {
      type: 'object',
      properties: {
        todos: {
          type: 'array',
          description: 'The complete checklist, in order',
          items: {
            type: 'object',
            properties: {
              content: { type: 'string', description: 'The step, imperative' },
              status: { type: 'string', enum: ['todo', 'doing', 'done'], description: 'todo = not started, doing = in progress (exactly one), done = finished' },
            },
            required: ['content', 'status'],
          },
        },
      },
      required: ['todos'],
    },
    execute: async function (a) {
      var list = (a && Array.isArray(a.todos)) ? a.todos : null;
      if (!list) throw new Error('todo_write: todos array required');
      if (list.length > 50) throw new Error('todo_write: max 50 items');
      var clean = [];
      for (var i = 0; i < list.length; i++) {
        var it = list[i] || {};
        var st = String(it.status || 'todo');
        if (st !== 'todo' && st !== 'doing' && st !== 'done') st = 'todo';
        var c = String(it.content || '').trim();
        if (c) clean.push({ content: c.slice(0, 200), status: st });
      }
      if (!clean.length) throw new Error('todo_write: at least one non-empty item');
      try { globalThis.__sofuu_todos = JSON.stringify(clean); } catch (eG) {}
      return 'checklist updated (' + clean.length + ' step' + (clean.length === 1 ? '' : 's') + ')';
    },
  };

  /* The "code" group — everything above. Opt out of the shell by listing
   * individual names instead, or by exporting SOFUU_NO_SHELL. S3: when the
   * shell is disabled the group (and every listing derived from it) omits
   * bash entirely; TOOLS.bash stays present so a direct call still gets the
   * explicit refusal at :execute instead of an opaque missing-function. */
  sofuu.tools.GROUP = ['read_file', 'write_file', 'edit_file', 'grep', 'glob',
                       'list_dir', 'bash', 'code_eval', 'todo_write']
    .filter(function (n) { return !(SHELL_DISABLED && n === 'bash'); });
  sofuu.tools.ENABLED = { bash: !SHELL_DISABLED };
})();
