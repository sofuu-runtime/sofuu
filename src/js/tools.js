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
//   - read/grep/glob are read-only and unrestricted (like cat/rg).
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
  var BASH_MAX_TIMEOUT_MS = 180000;
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

  /* Jail for mutating tools: the resolved path must stay under cwd. */
  function mustBeInRoot(p) {
    var abs = lexNorm(p);
    var root = lexNorm('.');
    if (abs !== root && abs.indexOf(root + '/') !== 0) {
      throw new Error('path escapes the project directory (writes are jailed to cwd): ' + p);
    }
    return abs;
  }

  function rel(p) {
    var abs = lexNorm(p);
    var root = lexNorm('.');
    if (abs === root) return '.';
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
      var p = mustBeInRoot(a.path);
      await mkdirp(parentDir(p));
      var existed = await sofuu.fs.exists(p);
      await sofuu.fs.writeFile(p, a.content);
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
      var p = mustBeInRoot(a && a.path);
      var oldS = String((a && a.old_string) == null ? '' : a.old_string);
      if (!oldS) throw new Error('edit_file: old_string is required (quote it exactly from read_file)');
      var newS = String((a && a.new_string) == null ? '' : a.new_string);
      var text = await sofuu.fs.readFile(p, 'utf8');
      var count = 0, idx = 0;
      while ((idx = text.indexOf(oldS, idx)) >= 0) { count++; idx += oldS.length; }
      if (count === 0) {
        throw new Error('edit_file: old_string not found in ' + rel(p) + ' — read_file it and copy the exact text');
      }
      if (count > 1 && !(a && a.replace_all)) {
        throw new Error('edit_file: old_string matches ' + count + ' places in ' + rel(p) +
          ' — include more surrounding context to make it unique, or pass replace_all:true');
      }
      var updated = (a && a.replace_all) ? text.split(oldS).join(newS) : text.replace(oldS, newS);
      await sofuu.fs.writeFile(p, updated);
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
      var gfilter = a && a.glob ? globToRe(a.glob) : null;
      var state = { files: 0, done: false };
      var hits = [];
      await walk(root, async function (file) {
        if (state.done) return;
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
      var state = { files: 0, done: false };
      var found = [];
      await walk(root, async function (file) {
        if (found.length >= GLOB_MAX) { state.done = true; return; }
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
    description: 'Run a shell command (/bin/sh -c) in the project directory and return stdout+stderr+exit code. Killed at the timeout (default 60s, max 180s). Prefer specific tools for reading/editing files.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to run' },
        timeout_ms: { type: 'number', description: 'Kill after this many ms (1000-180000, default 60000)' },
      },
      required: ['command'],
    },
    execute: async function (a) {
      if (process.env.SOFUU_NO_SHELL) throw new Error('bash: disabled (SOFUU_NO_SHELL is set)');
      var cmd = String((a && a.command) || '');
      if (!cmd.trim()) throw new Error('bash: command required');
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
  sofuu.tools.GROUP = ['read_file', 'write_file', 'edit_file', 'grep', 'glob', 'list_dir', 'bash'];
})();
