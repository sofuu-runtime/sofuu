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
//   - bash runs /bin/sh -c with a hard timeout (default 60s, max 180s) and
//     kills the child on expiry; stdout capped. A shell can still touch
//     anything the user can — set SOFUU_NO_SHELL=1 to strip bash entirely.
//   - read/grep/glob are read-only (like cat/rg) but refuse credential
//     stores — .sofuu/config.json, .ssh/*, *.pem|.key|.p12|.pfx (js-5:
//     grep/glob skip them so keys never reach the provider transcript).
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
    var root = lexNorm('.');    if (abs === root) return '.';
    if (abs.indexOf(root + '/') === 0) return abs.slice(root.length + 1);
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

  /* Recursive walk with caps. visit(file) is awaited per file. */
  async function walk(root, visit, state) {
    state = state || { files: 0, done: false };
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
        if (await isDir(full)) queue.push(full);
        else { state.files++; await visit(full); }
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
   * can still paste credentials manually if a task truly needs them. */
  function sensitiveReason(p) {
    var abs = lexNorm(p);
    var base = basename(abs);
    if (/(^|\/)\.sofuu\/config\.json$/.test(abs)) {
      return 'holds API keys — ask the user to provide what you need';
    }
    if (/(^|\/)\.ssh\//.test(abs + '/')) {
      return 'SSH keys are off-limits';
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
    description: 'Read a text file. Returns cat -n style numbered lines so exact text can be quoted into edit_file. For large files pass start/end (1-based, inclusive).',
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
      var p = lexNorm(a && a.path);
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
   * cannot flood the TUI (or the event payload): callers pass the limits. */
  var DIFF_MAX_BODY_LINES = 40;
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
    for (var l = from; l < to; l++) {
      if (l === startLine0) {
        /* The replaced block: old lines out, new lines in. */
        for (var o = 0; o < oldBlock.length; o++) merged.push('-' + oldBlock[o]);
        for (var nI = 0; nI < newBlock.length; nI++) merged.push('+' + newBlock[nI]);
        l += oldBlock.length - 1; /* skip the replaced originals */
        continue;
      }
      merged.push(' ' + allOld[l]);
    }
    var truncated = false;
    if (merged.length > DIFF_MAX_BODY_LINES) {
      merged = merged.slice(0, DIFF_MAX_BODY_LINES);
      truncated = true;
    }
    var out = ['--- a/' + rel + ' (line ' + startLine + ')',
               '+++ b/' + rel,
               '@@ -' + (from + 1) + ',' + (to - from) +
               ' +' + (from + 1) + ',' + (to - from + (newBlock.length - oldBlock.length)) + ' @@']
      .concat(merged);
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
    for (var i = 0; i < shown; i++) out.push('+' + lines[i]);
    if (lines.length > shown) out.push('... (' + (lines.length - shown) + ' more lines)');
    return out.join('\n');
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
      return (existed ? 'overwrote ' : 'created ') + rel(p) + ' (' + a.content.length + ' chars)';
    },
  };

  TOOLS.edit_file = {
    name: 'edit_file',
    description: 'Replace an exact string in a file. old_string must match exactly ONE place (quote it from read_file, including whitespace); add surrounding context if it is ambiguous, or set replace_all. Jailed to the project directory.',
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
      var text = await sofuu.fs.readFile(p, 'utf8');
      var count = 0, idx = 0, first = -1;
      while ((idx = text.indexOf(oldS, idx)) >= 0) { count++; if (first < 0) first = idx; idx += oldS.length; }
      if (count === 0) {
        throw new Error('edit_file: old_string not found in ' + rel(p) + ' — read_file it and copy the exact text');
      }
      if (count > 1 && !(a && a.replace_all)) {
        throw new Error('edit_file: old_string matches ' + count + ' places in ' + rel(p) +
          ' — include more surrounding context to make it unique, or pass replace_all:true');
      }
      /* js-6 (AUDIT-2026-09-07): splice literally — String.replace (the old
       * single-match branch) interprets $&, $$, $` and $' in new_string as
       * replacement patterns, silently corrupting the file when the new
       * text contains those two-char sequences. split/join (replace_all)
       * was already literal; the single-match branch now is too. */
      var updated = (a && a.replace_all) ? text.split(oldS).join(newS)
                                         : text.slice(0, first) + newS + text.slice(first + oldS.length);
      await sofuu.fs.writeFile(p, updated);
      setLastDiff('edit_file', buildEditDiff(rel(p), text, first, oldS, newS, count));
      return 'edited ' + rel(p) + ': ' + ((a && a.replace_all) ? count + ' replacements' : '1 replacement') +
        ' (' + (updated.length - text.length >= 0 ? '+' : '') + (updated.length - text.length) + ' chars)';
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
      var root = lexNorm((a && a.path) || '.');
      // Jail: searching outside the project (e.g. path '/') would walk the
      // whole filesystem — reject it like writes.
      var cwdRoot = lexNorm('.');
      if (root !== cwdRoot && root.indexOf(cwdRoot + '/') !== 0) {
        throw new Error('grep: path escapes the project directory (searches are jailed to cwd): ' + (a && a.path));
      }
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
      var root = lexNorm((a && a.path) || '.');
      var cwdRoot = lexNorm('.');
      if (root !== cwdRoot && root.indexOf(cwdRoot + '/') !== 0) {
        throw new Error('glob: path escapes the project directory (jailed to cwd): ' + (a && a.path));
      }
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
      var p = lexNorm((a && a.path) || '.');
      var cwdRoot = lexNorm('.');
      if (p !== cwdRoot && p.indexOf(cwdRoot + '/') !== 0) {
        throw new Error('list_dir: path escapes the project directory (jailed to cwd): ' + (a && a.path));
      }
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
        var proc;
        try {
          proc = sofuu.spawn({
            command: '/bin/sh',
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

  sofuu.tools._loaded = true;
  sofuu.tools.VERSION = '1';
  sofuu.tools.TOOLS = TOOLS;
  /* The "code" group — everything above. Opt out of the shell by listing
   * individual names instead, or by exporting SOFUU_NO_SHELL. */
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

  sofuu.tools.GROUP = ['read_file', 'write_file', 'edit_file', 'grep', 'glob', 'list_dir', 'bash', 'todo_write'];
})();
