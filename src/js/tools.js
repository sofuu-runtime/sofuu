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

  /* R2.3 (SPEC T2.3): live tool subprocess registry. bash/code_eval register
   * their spawn handle here so a run cancel can kill an in-flight tool —
   * withCancelGuard stops the loop, this stops the child. */
  var LIVE_PROCS = [];
  function trackProc(handle) {
    if (!handle) return handle;
    LIVE_PROCS.push(handle);
    return handle;
  }
  function untrackProc(handle) {
    var i = LIVE_PROCS.indexOf(handle);
    if (i >= 0) LIVE_PROCS.splice(i, 1);
  }

  /* S4 (2026-10-23): the shared policy floor (SPEC R7). Every executor in
   * this driver consults the same evaluator the agent loop uses, so a host
   * calling a tool directly cannot bypass a project/user deny. */
  function policyDecision(op, fields) {
    try {
      if (typeof sofuu === 'undefined' || !sofuu.policy ||
          typeof sofuu.policy.evaluate !== 'function') return null;
      var req = { surface: 'js', op: op };
      if (fields) {
        for (var k in fields) {
          if (Object.prototype.hasOwnProperty.call(fields, k) && fields[k] !== undefined) {
            req[k] = fields[k];
          }
        }
      }
      var raw = sofuu.policy.evaluate(req);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }
  function policyCheck(op, fields) {
    var d = policyDecision(op, fields);
    if (d && d.decision === 'deny') {
      throw new Error(op + ': denied by policy (' + d.rule + '): ' + (d.reason || ''));
    }
  }

  /* ── H-01 (AUDIT-2026-10-29): ONE enforcement point ──────────────────
   *
   * Before this, policyCheck('tool.run', …) ran in exactly ONE executor (the
   * bash tool above), so the other twelve ignored tools.deny entirely and a
   * caller that reached an executor directly — the agent loop builds its own
   * dispatch, host code can call TOOLS.read_file.execute(), inline/toolpack
   * tools run in the same table — bypassed the policy the inventory
   * (sofuu.tools.list) and the loop both reported. Tools are a policy surface
   * (R7.6), so the check belongs to the TABLE, not to each tool's body.
   *
   * executeWithPolicy is now the only way in: every spec.execute is replaced
   * at load (installPolicyWrappers, end of this file) with a wrapper that
   * asks the SAME evaluator list() asks, through the SAME request builder, so
   * "listed as denied" and "denied at run time" cannot diverge.
   *
   * Double-run, and why it is safe: agent.js evaluates the policy BEFORE
   * approval — it must, because an `ask` verdict is what decides whether the
   * host gets prompted. The wrapper evaluates again on the way in. That is
   * safe by construction rather than by caching: sofuu.policy.evaluate is a
   * PURE function of (request, policy) with no side effects, so the second
   * answer is the first answer; and a loop-level deny never reaches dispatch,
   * so the wrapper can only ever see allow/ask here. The loop was deliberately
   * NOT rewritten to call the wrapper: its check also runs before approval.mode
   * handling and lives in a file outside this change's scope.
   *
   * `ask` is passed THROUGH the wrapper: approval is the caller's job (the
   * loop prompts, a host may run unattended). The wrapper never invents an
   * approval UI and never turns `ask` into a silent allow of a DENY. */
  var POLICY_ARG_KEYS = ['command', 'path', 'cwd', 'host', 'url', 'server'];
  var POLICY_ARG_MAX = 8192;

  /* The request shape, in ONE place so the listing and the executor cannot
   * disagree: `tool` at the top level (what tools.deny/tools.allow match) plus
   * the parameter projection both as `params` (the documented shape) and
   * flattened, because the evaluator's shell/net/fs rules read the top level
   * (`command`, `path`, `host`). Only scalars ride along, capped: serializing
   * a write_file body into an evaluator request would cost a JSON round-trip
   * per call and could not change the decision. */
  function toolPolicyRequest(name, params) {
    var req = { op: 'tool.run', tool: name };
    if (params && typeof params === 'object') {
      var proj = null;
      for (var i = 0; i < POLICY_ARG_KEYS.length; i++) {
        var k = POLICY_ARG_KEYS[i];
        var v = params[k];
        if (v === undefined || v === null) continue;
        if (typeof v !== 'string' && typeof v !== 'number') continue;
        if (proj === null) proj = {};
        proj[k] = String(v).slice(0, POLICY_ARG_MAX);
      }
      if (proj !== null) {
        req.params = proj;
        for (var pk in proj) {
          if (Object.prototype.hasOwnProperty.call(proj, pk)) req[pk] = proj[pk];
        }
      }
    }
    return req;
  }

  /* Returns the parsed decision, or null when there is no evaluator (a
   * stripped runtime) — null means "no opinion", never "allow". */
  function toolPolicyDecision(name, params) {
    try {
      if (typeof sofuu === 'undefined' || !sofuu.policy ||
          typeof sofuu.policy.evaluate !== 'function') return null;
      var raw = sofuu.policy.evaluate(toolPolicyRequest(name, params));
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  /* The typed denial H-01 asks for, carrying the R7.3 rule id in the message
   * AND as properties so a driver can branch without parsing text. The shape
   * keeps the pre-existing "denied by policy (<rule>)" substring the contract
   * harnesses assert on (t72/t75, tests/policy_test.js). */
  function policyDeniedError(name, d) {
    var rule = (d && d.rule) || 'R7/default';
    var reason = (d && d.reason) || '';
    var err = new Error('E_POLICY_DENIED: Tool \'' + name + '\' is denied by policy (' +
      rule + ')' + (reason ? ': ' + reason : ''));
    err.code = 'E_POLICY_DENIED';
    err.decision = 'deny';
    err.rule = rule;
    err.tool = name;
    if (reason) err.policy_reason = reason;
    return err;
  }

  /* H-02(a) bridge: while a policy-governed dispatch is in flight, the native
   * layers can tell "the runtime is acting for a model/remote caller" from "a
   * host script is doing its own plumbing" (process_spawn's gate reads this
   * depth). It is a depth, not a boolean, because a tool may dispatch another
   * (apply_patch → bash style); it is clamped at 0 so a stray decrement can
   * never leave the marker stuck on. */
  var TOOL_ACTIVE = 0;
  function toolDispatchBegin() {
    TOOL_ACTIVE++;
    try { globalThis.__sofuu_tool_active = TOOL_ACTIVE; } catch (e) {}
  }
  function toolDispatchEnd() {
    TOOL_ACTIVE = TOOL_ACTIVE > 0 ? TOOL_ACTIVE - 1 : 0;
    try { globalThis.__sofuu_tool_active = TOOL_ACTIVE; } catch (e) {}
  }

  /* The single enforcement point. Works when copied off the object and called
   * unbound (`const f = TOOLS.edit_file.execute; f(...)`), which a host is
   * allowed to do — no `this`. */
  function executeWithPolicy(toolName, params, ctx) {
    var name = String(toolName == null ? '' : toolName);
    var spec = TOOLS[name];
    if (!spec || typeof spec.__exec !== 'function') {
      var e = new Error('E_TOOL: unknown tool \'' + name + '\'');
      e.code = 'E_TOOL';
      e.tool = name;
      throw e;
    }
    var d = toolPolicyDecision(name, params);
    if (d && d.decision === 'deny') throw policyDeniedError(name, d);
    toolDispatchBegin();
    var out;
    try {
      out = spec.__exec(params, ctx);
    } catch (err) {
      toolDispatchEnd();
      throw err;
    }
    if (out && typeof out.then === 'function') {
      return out.then(function (v) { toolDispatchEnd(); return v; },
                      function (err) { toolDispatchEnd(); throw err; });
    }
    toolDispatchEnd();
    return out;
  }

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
        policyCheck('fs.read', { path: abs });
        return abs;
      }
      if (targetReal !== rootRealPath && targetReal.indexOf(rootRealPath + '/') !== 0) {
        throw new Error(who + ': path resolves (via symlink) outside the project directory: ' + p +
          ' → ' + targetReal);
      }
      policyCheck('fs.read', { path: targetReal });
      return targetReal;
    }
    policyCheck('fs.read', { path: abs });
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
          policyCheck('fs.write', { path: canonical });
          return canonical;
        }
        /* targetReal === null: no existing ancestor within reach — the
         * lexical check above already ran, nothing more to verify. */
      }
    }
    policyCheck('fs.write', { path: abs });
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
      if (process.env.SOFUU_NO_SHELL) {
        /* Same typed refusal the native spawn gate returns (H-02a), so the
         * tool and sofuu.spawn cannot disagree about the operator's kill
         * switch — and "disabled" stays in the text for the inventory
         * harness. */
        var eNoShell = new Error('E_PERMISSION_DENIED: Process execution disabled by policy');
        eNoShell.code = 'E_PERMISSION_DENIED';
        eNoShell.rule = 'R7.6/shell.enabled';
        eNoShell.tool = 'bash';
        throw eNoShell;
      }
      var cmd = String((a && a.command) || '');
      if (!cmd.trim()) throw new Error('bash: command required');
      /* S4/H-01: the policy check for this call now runs in the wrapper
       * (executeWithPolicy) that every executor sits behind, with the same
       * request `{op:'tool.run', tool:'bash', command}`. Keeping a second
       * copy here would evaluate the same pure decision twice per call for
       * no extra safety — the wrapper cannot be bypassed. */
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
      return await new Promise(function (resolve, rejectP) {
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
          proc = trackProc(sofuu.spawn({
            command: shCmd,
            args: shArgs,
            onStdout: function (chunk) { capAdd(String(chunk)); },
            onStderr: function (chunk) { capErr(String(chunk)); },
            onExit: function (c) { untrackProc(proc); code = c; finish(''); },
          }));
        } catch (e) {
          /* H-01/H-02(a): a policy refusal must not be reported as a normal
           * shell result. The wrapper already refuses the common cases; the
           * native spawn gate can still refuse here (shell.allow at the spawn
           * layer, the SOFUU_NO_SHELL kill switch), and that has to surface
           * as a tool ERROR — a model must not read a denial as `[exit 0]`. */
          var emsg = String((e && e.message) || e);
          if (emsg.indexOf('E_PERMISSION_DENIED') >= 0 || emsg.indexOf('denied by policy') >= 0) {
            done = true;
            if (tid !== null) clearTimeout(tid);
            rejectP(e);
            return;
          }
          resolve('$ ' + cmd + '\nbash: spawn failed: ' + emsg);
          return;
        }
        tid = setTimeout(function () {
          try { proc.kill(); } catch (e) {}
          finish('\n[killed: timed out after ' + timeout + 'ms]\n');
        }, timeout);
      });
    },
  };

  /* ── T4 (2026-10-23): apply_patch — Codex-style multi-file patch ──
   * *** Begin Patch / Add File / Update File (@@ hunks) / Delete File /
   * *** End Patch. Every operation is parsed and applied IN MEMORY first;
   * any bad hunk or syntax rejection aborts before the first write, so a
   * multi-file patch is atomic on the filesystem. Context lines must match
   * exactly one place per hunk (whitespace-insensitive, same matcher as
   * edit_file). One diff summary rides the tool event. */
  function parseApplyPatch(text) {
    var lines = String(text).replace(/\r\n/g, '\n').split('\n');
    var i = 0;
    while (i < lines.length && lines[i].trim() === '') i++;
    if (!/^\*\*\* Begin Patch\s*$/.test(lines[i] || '')) {
      throw new Error('apply_patch: missing "*** Begin Patch"');
    }
    i++;
    var ops = [];
    while (i < lines.length) {
      var l = lines[i];
      if (/^\*\*\* End Patch\s*$/.test(l)) return ops;
      var mAdd = /^\*\*\* Add File: (.+)$/.exec(l);
      var mDel = /^\*\*\* Delete File: (.+)$/.exec(l);
      var mUpd = /^\*\*\* Update File: (.+)$/.exec(l);
      if (mAdd) {
        i++;
        var content = [];
        while (i < lines.length && !/^\*\*\* /.test(lines[i])) {
          if (lines[i].charAt(0) !== '+') {
            throw new Error('apply_patch: add-file lines must start with "+" (got "' + lines[i].slice(0, 40) + '")');
          }
          content.push(lines[i].slice(1));
          i++;
        }
        ops.push({ kind: 'add', path: mAdd[1].trim(),
                   content: content.join('\n') + (content.length ? '\n' : '') });
        continue;
      }
      if (mDel) { i++; ops.push({ kind: 'delete', path: mDel[1].trim() }); continue; }
      if (mUpd) {
        i++;
        var hunks = [], h = null;
        while (i < lines.length && !/^\*\*\* /.test(lines[i])) {
          var ln = lines[i];
          if (/^@@/.test(ln)) {
            if (h && (h.old.length || h.new.length)) hunks.push(h);
            h = { header: ln, old: [], new: [] };
            i++;
            continue;
          }
          if (!h) { i++; continue; } /* preamble between the header and @@ */
          var c = ln.charAt(0);
          if (c === '-') h.old.push(ln.slice(1));
          else if (c === '+') h.new.push(ln.slice(1));
          else if (c === ' ' || ln === '') { h.old.push(ln.slice(1)); h.new.push(ln.slice(1)); }
          else throw new Error('apply_patch: unexpected hunk line "' + ln.slice(0, 40) + '"');
          i++;
        }
        if (h && (h.old.length || h.new.length)) hunks.push(h);
        if (!hunks.length) throw new Error('apply_patch: update for ' + mUpd[1].trim() + ' has no hunks');
        ops.push({ kind: 'update', path: mUpd[1].trim(), hunks: hunks });
        continue;
      }
      if (l.trim() === '') { i++; continue; }
      throw new Error('apply_patch: unexpected line "' + l.slice(0, 60) + '"');
    }
    throw new Error('apply_patch: missing "*** End Patch"');
  }
  function applyPatchHunks(text, hunks) {
    var lines = String(text).split('\n');
    for (var h = 0; h < hunks.length; h++) {
      var want = hunks[h].old;
      if (!want.length) {
        throw new Error('apply_patch: hunk ' + (h + 1) + ' has no context; include surrounding lines');
      }
      var hits = [];
      for (var st = 0; st + want.length <= lines.length; st++) {
        var ok = true;
        for (var k = 0; k < want.length; k++) {
          if (normLine(lines[st + k]) !== normLine(want[k])) { ok = false; break; }
        }
        if (ok) hits.push(st);
      }
      if (hits.length !== 1) {
        throw new Error('apply_patch: hunk ' + (h + 1) + ' context matched ' + hits.length +
                        ' place' + (hits.length === 1 ? '' : 's') + ' (must be unique)');
      }
      lines.splice.apply(lines, [hits[0], want.length].concat(hunks[h].new));
    }
    return lines.join('\n');
  }

  TOOLS.apply_patch = {
    name: 'apply_patch',
    description: 'Apply a multi-file patch atomically. Format: "*** Begin Patch" then "*** Add File: path" (+lines), "*** Update File: path" with @@ hunks (context lines start with a space, removed with "-", added with "+"), or "*** Delete File: path", then "*** End Patch". Hunk context must be unique (quote enough lines). .js/.ts/.json results are syntax-checked; any failure rejects the WHOLE patch and nothing is written. Jailed to the project directory.',
    parameters: {
      type: 'object',
      properties: {
        patch: { type: 'string', description: 'The complete *** Begin Patch … *** End Patch text' },
      },
      required: ['patch'],
    },
    execute: async function (a) {
      var text = (a && typeof a.patch === 'string') ? a.patch : '';
      if (!text.trim()) throw new Error('apply_patch: patch text required');
      var ops = parseApplyPatch(text);
      if (!ops.length) throw new Error('apply_patch: no file operations in the patch');
      var planned = [], i;
      /* Plan phase: resolve every path, apply every hunk in memory, gate the
       * syntax. Nothing touches disk until the whole patch is valid. */
      for (i = 0; i < ops.length; i++) {
        var op = ops[i];
        var p = await mustBeInRoot(op.path);
        if (op.kind === 'add') {
          if (await sofuu.fs.exists(p)) {
            throw new Error('apply_patch: add file already exists: ' + op.path);
          }
          var gateA = syntaxVerdict(op.content, p);
          if (gateA && gateA.ok === false) {
            throw new Error('apply_patch: syntax rejected ' + rel(p) +
              (gateA.line > 0 ? ':' + gateA.line : '') + ': ' + gateA.error + ' — nothing was written');
          }
          planned.push({ kind: 'add', path: p, content: op.content });
        } else if (op.kind === 'delete') {
          var beforeD = await sofuu.fs.readFile(p, 'utf8');
          planned.push({ kind: 'delete', path: p, before: beforeD });
        } else {
          var beforeU = await sofuu.fs.readFile(p, 'utf8');
          var afterU = applyPatchHunks(beforeU, op.hunks);
          try { syntaxGate(beforeU, afterU, p); } catch (eG) {
            throw new Error(String(eG.message || eG).replace(/^edit_file:/, 'apply_patch:') + ' (nothing was written)');
          }
          planned.push({ kind: 'update', path: p, after: afterU, hunks: op.hunks.length });
        }
      }
      /* Commit phase. */
      var out = [], diffParts = [];
      for (i = 0; i < planned.length; i++) {
        var pl = planned[i];
        if (pl.kind === 'add') {
          await mkdirp(parentDir(pl.path));
          await sofuu.fs.writeFile(pl.path, pl.content);
          diffParts.push('+++ ' + rel(pl.path) + ' (new file, ' + pl.content.split('\n').length + ' lines)');
          out.push('added ' + rel(pl.path));
        } else if (pl.kind === 'update') {
          await sofuu.fs.writeFile(pl.path, pl.after);
          diffParts.push(buildWritePreview(rel(pl.path), pl.after));
          out.push('updated ' + rel(pl.path) + ' (' + pl.hunks + ' hunk' + (pl.hunks === 1 ? '' : 's') + ')');
        } else {
          await sofuu.fs.rm(pl.path);
          diffParts.push('--- ' + rel(pl.path) + ' (deleted)');
          out.push('deleted ' + rel(pl.path));
        }
      }
      setLastDiff('apply_patch', diffParts.join('\n\n'));
      var res = 'apply_patch: ' + out.join('; ');
      /* Post-edit context for the first updated/added file (T3.1 rider). */
      for (i = 0; i < planned.length; i++) {
        if (planned[i].kind === 'update') {
          try { res += anchorRider(planned[i].after, 0, 2); } catch (eR) {}
          break;
        }
      }
      return res;
    },
  };

  /* code_eval — In-process deterministic JavaScript/TypeScript sandboxed execution.
   * Powered by sofuu.sandbox. Has zero-subprocess overhead (< 1ms startup),
   * strict instruction fuel limit (500k ops), 2000ms wall-clock deadline,
   * 16MB memory ceiling, captured console logs, and hermetic host isolation. */
  /* T4 (2026-10-23): persistent sandbox realms for code_eval. Keyed by an
   * explicit session_id or by the run scope, so two concurrent runs never
   * share helpers. endRunScope() closes everything a run left behind. */
  var CODE_SESSIONS = {};
  function codeSessionKey(a) {
    if (a && typeof a.session_id === 'string' && a.session_id) return 'id:' + a.session_id;
    if (a && a.persist === true) {
      var sc = (typeof a.__scope === 'string' && a.__scope) ? a.__scope : 'default';
      return 'run:' + sc.replace(/[^A-Za-z0-9_-]/g, '');
    }
    return '';
  }
  sofuu.tools.endRunScope = function (scope) {
    var sc = String(scope || '');
    var prefix = 'run:' + sc.replace(/[^A-Za-z0-9_-]/g, '');
    Object.keys(CODE_SESSIONS).forEach(function (k) {
      if (k === prefix) {
        try { CODE_SESSIONS[k].close(); } catch (e) {}
        delete CODE_SESSIONS[k];
      }
    });
    try {
      delete globalThis['__sofuu_todos:' + sc.replace(/[^A-Za-z0-9_-]/g, '')];
    } catch (e2) {}
  };

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
        session_id: {
          type: 'string',
          description: 'Optional session name. Calls with the same id share one sandbox realm (helpers and globals persist).',
        },
        persist: {
          type: 'boolean',
          description: 'Keep one realm for this run even without a session_id (the default is a fresh realm per call).',
        },
        reset: {
          type: 'boolean',
          description: 'Drop the session realm before running this call.',
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

      var key = codeSessionKey(args);
      var res;
      if (key) {
        if (args && args.reset === true && CODE_SESSIONS[key]) {
          try { CODE_SESSIONS[key].close(); } catch (eRst) {}
          delete CODE_SESSIONS[key];
        }
        var sess = CODE_SESSIONS[key];
        if (!sess) {
          sess = sofuu.sandbox.create({
            fuel: fuel, timeoutMs: timeout, maxMemoryMB: 16,
            captureConsole: true, throwOnError: false,
          });
          CODE_SESSIONS[key] = sess;
        }
        /* `await`: a run only returns a promise when it parks on an async
         * host bridge; code_eval installs none, so this is the sync path. */
        res = await sess.run(code, { fuel: fuel, timeoutMs: timeout, captureConsole: true, throwOnError: false });
      } else {
        res = await sofuu.sandbox.run(code, {
          fuel: fuel,
          timeoutMs: timeout,
          maxMemoryMB: 16,
          captureConsole: true,
          throwOnError: false,
        });
      }

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
      /* T4: state is per run (the shared loop injects __scope), so a run
       * that never writes a checklist cannot inherit the previous one. */
      var scope = (a && typeof a.__scope === 'string' && a.__scope)
        ? ':' + a.__scope.replace(/[^A-Za-z0-9_-]/g, '') : '';
      try { globalThis['__sofuu_todos' + scope] = JSON.stringify(clean); } catch (eG) {}
      return 'checklist updated (' + clean.length + ' step' + (clean.length === 1 ? '' : 's') + ')';
    },
  };

  /* General-purpose system & hardware info tool. */
  TOOLS.system_info = {
    name: 'system_info',
    description: 'Get information about the runtime system, OS platform, hardware architecture, capabilities, and engine status.',
    parameters: {
      type: 'object',
      properties: {},
    },
    execute: async function () {
      var caps = {};
      try {
        if (typeof sofuu.capabilities === 'function') {
          caps = JSON.parse(sofuu.capabilities());
        }
      } catch (eCaps) {}
      var plat = 'unknown';
      try {
        plat = (typeof process !== 'undefined' && process.platform) || 'unknown';
      } catch (ePlat) {}
      var memInfo = null;
      try {
        if (typeof process !== 'undefined' && typeof process.memoryUsage === 'function') {
          var mu = process.memoryUsage();
          memInfo = { rss_mb: Math.round(mu.rss / 1048576) };
        }
      } catch (eMem) {}
      var info = {
        runtime: 'sofuu',
        version: (sofuu.VERSION || '0.3.0-beta'),
        platform: plat,
        capabilities: caps,
        memory: memInfo,
      };
      return JSON.stringify(info, null, 2);
    },
  };

  /* General-purpose vector memory retrieval tool (backed by QTSQ Cognitive Vector Memory). */
  TOOLS.memory_search = {
    name: 'memory_search',
    description: 'Search long-term semantic memory and knowledge base for stored facts, preferences, or past interactions.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query to match in semantic memory' },
        count: { type: 'number', description: 'Maximum number of memories to return (1-20, default 5)' },
      },
      required: ['query'],
    },
    execute: async function (args) {
      var q = String((args && args.query) || '').trim();
      if (!q) return 'error: empty query';
      var topK = Math.max(1, Math.min(20, (args && args.count) || 5));
      if (!sofuu.memory || typeof sofuu.memory.search !== 'function') {
        return '(no vector memory subsystem available)';
      }
      try {
        var hits = sofuu.memory.search({ query: q, top_k: topK });
        if (!hits || !hits.length) return '(no matching memories found)';
        return hits.map(function (h, i) {
          return (i + 1) + '. ' + (h.text || h.fact || JSON.stringify(h));
        }).join('\n');
      } catch (e) {
        return 'memory search error: ' + (e.message || String(e));
      }
    },
  };

  /* General-purpose vector memory storage tool. */
  TOOLS.memory_remember = {
    name: 'memory_remember',
    description: 'Store a durable fact, preference, or piece of information in long-term vector memory for future recall.',
    parameters: {
      type: 'object',
      properties: {
        fact: { type: 'string', description: 'The durable fact, preference, or information to remember' },
      },
      required: ['fact'],
    },
    execute: async function (args) {
      var fact = String((args && args.fact) || '').trim();
      if (!fact) return 'error: empty fact';
      if (!sofuu.memory || typeof sofuu.memory.learn !== 'function') {
        return '(no vector memory subsystem available)';
      }
      try {
        var id = sofuu.memory.learn(fact);
        return 'saved to long-term memory (id: ' + id + ')';
      } catch (e) {
        return 'memory remember error: ' + (e.message || String(e));
      }
    },
  };

  /* ── S6 (2026-10-23): one MCP config loader ──────────────────────
   * v2 entry: {name, command, args?, env?, cwd?, enabled?, tools?{allow,deny}}
   * Backward compatible with {name, command}. SOFUU_MCP_CONFIG overrides the
   * path (tests, project-local configs); otherwise ~/.sofuu/mcp.json. A
   * missing file is "no servers", a corrupt one is {ok:false} and must be
   * surfaced — never silently rewritten. */
  sofuu.mcp.loadConfig = async function (opts) {
    opts = opts || {};
    var path = opts.path ? String(opts.path) : '';
    if (!path) {
      try {
        path = (typeof process !== 'undefined' && process.env && process.env.SOFUU_MCP_CONFIG) || '';
      } catch (e0) {}
    }
    var home = '';
    try { home = (typeof process !== 'undefined' && process.env && process.env.HOME) || ''; } catch (eH) {}
    if (!path && home) path = home + '/.sofuu/mcp.json';
    if (!path) return { ok: true, path: '', servers: [] };

    var raw = null;
    try { raw = await sofuu.fs.readFile(path, 'utf8'); }
    catch (e) {
      /* ENOENT and friends: no config is a normal state. */
      return { ok: true, path: path, servers: [] };
    }
    var list;
    try { list = JSON.parse(raw); }
    catch (eP) {
      return { ok: false, path: path, error: 'mcp config is corrupt (left in place)', servers: [] };
    }
    if (!Array.isArray(list)) {
      return { ok: false, path: path, error: 'mcp config must be a JSON array', servers: [] };
    }
    var servers = [];
    for (var i = 0; i < list.length; i++) {
      var s = list[i] || {};
      if (s.enabled === false) continue;
      var command = String(s.command || '');
      if (!command) continue;
      var entry = { name: String(s.name || ('mcp' + (i + 1))), command: command };
      if (Array.isArray(s.args)) entry.args = s.args.map(String);
      if (s.env && typeof s.env === 'object') {
        var env = {};
        for (var k in s.env) {
          if (Object.prototype.hasOwnProperty.call(s.env, k)) {
            var v = s.env[k];
            if (v === null) {
              /* null removes an inherited allowlisted variable. */
              env[k] = null;
            } else if (v && typeof v === 'object' && typeof v.secret === 'string') {
              /* S6/R7.5: configs carry only the NAME; the value is resolved
               * at connect time by policy.secret() and never stored. */
              env[k] = '${secret:' + v.secret + '}';
            } else {
              env[k] = String(v);
            }
          }
        }
        entry.env = env;
      }
      if (typeof s.cwd === 'string' && s.cwd) entry.cwd = s.cwd;
      if (s.tools && typeof s.tools === 'object') {
        entry.tools = {
          allow: Array.isArray(s.tools.allow) ? s.tools.allow.map(String) : null,
          deny: Array.isArray(s.tools.deny) ? s.tools.deny.map(String) : null,
        };
      }
      servers.push(entry);
    }
    return { ok: true, path: path, servers: servers };
  };

  /* ── M1 (2026-10-23): paged MCP lists ─────────────────────────────
   * MCP list methods are cursor-based; the native client exposes one page
   * (`client.call`). Wrapping connect() gives every consumer (agent, chat,
   * plain scripts) the whole inventory without a native async state
   * machine. A repeated cursor or >32 pages stops, so a hostile server
   * cannot loop its caller forever. */
  (function () {
    if (!sofuu.mcp || typeof sofuu.mcp.connect !== 'function' || sofuu.mcp.__paged) return;
    var realConnect = sofuu.mcp.connect;
    function all(client, method, key) {
      var out = [];
      var cursor = null;
      var pages = 0;
      function step() {
        var params = cursor ? { cursor: cursor } : {};
        return Promise.resolve(client.call(method, params)).then(function (res) {
          var arr = (res && res[key]) || [];
          for (var i = 0; i < arr.length; i++) out.push(arr[i]);
          var next = res && res.nextCursor;
          pages++;
          if (next && next !== cursor && pages < 32) {
            cursor = next;
            return step();
          }
          var r = {};
          r[key] = out;
          return r;
        });
      }
      return step();
    }
    /* S6/R7.5: "${secret:NAME}" env values are resolved HERE, at connect
     * time, through policy.secret(). Configs and loadConfig() only ever see
     * the name; an unresolved name is a hard error (never spawn with the
     * literal sentinel). */
    function resolveSecretEnv(env) {
      if (!env || typeof env !== 'object') return env;
      var out = {};
      for (var k in env) {
        if (!Object.prototype.hasOwnProperty.call(env, k)) continue;
        var v = env[k];
        if (v === null) { out[k] = null; continue; }
        var m = (typeof v === 'string') ? /^\$\{secret:([A-Za-z0-9_.-]+)\}$/.exec(v) : null;
        if (!m) { out[k] = String(v); continue; }
        if (!sofuu.policy || typeof sofuu.policy.secret !== 'function') {
          throw new Error('mcp.connect: env ' + k + ' needs a secret but sofuu.policy.secret is unavailable');
        }
        out[k] = sofuu.policy.secret(m[1]);
      }
      return out;
    }
    /* S5/M1: the policy's default MCP call deadline. Applied only when the
     * caller passed no timeout, so an explicit {timeoutMs} always wins. */
    function policyCallTimeout() {
      try {
        if (!sofuu.policy || typeof sofuu.policy.get !== 'function') return 0;
        var p = JSON.parse(sofuu.policy.get());
        var ms = p && p.mcp && p.mcp.callTimeoutMs;
        if (typeof ms === 'number' && ms > 0) return Math.min(600000, Math.max(50, ms));
      } catch (e) {}
      return 0;
    }
    /* ── M2 (2026-10-23): Streamable HTTP transport (JS) ─────────────
     * One POST per JSON-RPC message; a response may be JSON or an SSE
     * stream carrying the reply (and any interleaved server traffic).
     * `Mcp-Session-Id` is echoed after initialize and DELETEd on
     * disconnect. Legacy HTTP+SSE is an explicit typed refusal for now
     * (the risk register's "ship Streamable HTTP first" call). */
    function httpHostOf(url) {
      var h = String(url || '').replace(/^[a-z][a-z0-9+.-]*:\/\//i, '');
      var slash = h.indexOf('/'); if (slash >= 0) h = h.slice(0, slash);
      var q = h.indexOf('?'); if (q >= 0) h = h.slice(0, q);
      var at = h.lastIndexOf('@'); if (at >= 0) h = h.slice(at + 1);
      if (h.charAt(0) === '[') { var rb = h.indexOf(']'); if (rb > 0) return h.slice(1, rb); }
      var colon = h.lastIndexOf(':'); if (colon > 0) h = h.slice(0, colon);
      return h;
    }
    function mcpFetch(url, init) {
      var f = (typeof fetch === 'function') ? fetch : (sofuu.fetch || null);
      if (!f) throw new Error('mcp http: fetch is unavailable in this build');
      return f(url, init);
    }
    sofuu.mcp._connectHttp = function (cfg) {
      cfg = cfg || {};
      var url = String(cfg.url || '');
      if (!url) return Promise.reject(new Error('mcp.connect: url required for the HTTP transport'));
      var host = httpHostOf(url);
      var serverId = cfg.name ? String(cfg.name) : url;
      var pdNet = policyDecision('net.fetch', { host: host });
      if (pdNet && pdNet.decision === 'deny') {
        return Promise.reject(new Error('mcp.connect: denied by policy (' + pdNet.rule + '): ' + (pdNet.reason || host)));
      }
      var baseHeaders = { 'content-type': 'application/json', 'accept': 'application/json, text/event-stream' };
      if (cfg.headers && typeof cfg.headers === 'object') {
        for (var hk in cfg.headers) {
          if (Object.prototype.hasOwnProperty.call(cfg.headers, hk)) baseHeaders[hk.toLowerCase()] = String(cfg.headers[hk]);
        }
      }
      var sessionId = '';
      var nextId = 1;
      var pending = {};
      var noteCb = null;
      var protocol = '2025-06-18';
      var closed = false;

      function wireHeaders() {
        var h = {};
        for (var k in baseHeaders) h[k] = baseHeaders[k];
        if (sessionId) h['mcp-session-id'] = sessionId;
        return h;
      }
      function postMessage(msg) {
        return mcpFetch(url, { method: 'POST', headers: wireHeaders(), body: JSON.stringify(msg) });
      }
      function settle(id, ok, value) {
        var p = pending[id];
        if (!p) return;
        delete pending[id];
        if (p.timer) clearTimeout(p.timer);
        if (ok) p.resolve(value); else p.reject(value);
      }
      function handleMessage(msg) {
        if (!msg || msg.jsonrpc !== '2.0') return;
        if (msg.id !== undefined && !msg.method) {
          var v = (msg.result === undefined) ? null : msg.result;
          if (msg.error) {
            settle(msg.id, false, new Error(String((msg.error && msg.error.message) || ('MCP error ' + (msg.error && msg.error.code)))));
          } else settle(msg.id, true, v);
          return;
        }
        if (msg.method && msg.id !== undefined) {
          var reply = { jsonrpc: '2.0', id: msg.id };
          if (msg.method === 'ping') reply.result = {};
          else if (msg.method === 'roots/list') {
            var roots = [];
            try {
              var pol = JSON.parse(sofuu.policy.get());
              if (pol && pol.fs && Array.isArray(pol.fs.roots)) roots = pol.fs.roots;
            } catch (eR) {}
            reply.result = { roots: roots.map(function (p) { return { uri: 'file://' + p }; }) };
          } else reply.error = { code: -32601, message: 'Method not found' };
          postMessage(reply).then(consumeResponse, function () {});
          return;
        }
        if (msg.method && noteCb) {
          try { noteCb(String(msg.method), JSON.stringify(msg.params === undefined ? null : msg.params)); } catch (eN) {}
        }
      }
      async function consumeResponse(res) {
        try {
          var sid = res.headers && res.headers.get ? res.headers.get('mcp-session-id') : '';
          if (sid) sessionId = String(sid);
        } catch (eH) {}
        if (res.status === 202 || res.status === 204) return;
        if (!res.ok && res.status >= 400) {
          var bodyText = '';
          try { bodyText = await res.text(); } catch (eT) {}
          throw new Error('mcp http: HTTP ' + res.status + (bodyText ? (': ' + bodyText.slice(0, 200)) : ''));
        }
        var ct = '';
        try { ct = String((res.headers && res.headers.get && res.headers.get('content-type')) || ''); } catch (eC) {}
        if (ct.indexOf('text/event-stream') >= 0) { await readSseStream(res); return; }
        var text = await res.text();
        if (!text || !text.trim()) return;
        var one;
        try { one = JSON.parse(text); } catch (eP) { throw new Error('mcp http: response is not JSON'); }
        if (Array.isArray(one)) { for (var i = 0; i < one.length; i++) handleMessage(one[i]); }
        else handleMessage(one);
      }
      async function readSseStream(res) {
        var it = res.body[Symbol.asyncIterator]();
        var buf = '';
        for (;;) {
          var step = await it.next();
          if (step.done) break;
          var chunk = '';
          var v = step.value;
          if (typeof v === 'string') chunk = v;
          else { for (var b = 0; b < v.length; b++) chunk += String.fromCharCode(v[b]); }
          buf += chunk;
          for (;;) {
            var nl = buf.indexOf('\n');
            if (nl < 0) break;
            var line = buf.slice(0, nl).replace(/\r$/, '');
            buf = buf.slice(nl + 1);
            if (!line || line.charAt(0) === ':') continue;
            if (line.indexOf('data:') === 0) {
              var data = line.slice(5);
              if (data.charAt(0) === ' ') data = data.slice(1);
              var parsed;
              try { parsed = JSON.parse(data); } catch (eD) { continue; }
              if (Array.isArray(parsed)) { for (var j = 0; j < parsed.length; j++) handleMessage(parsed[j]); }
              else handleMessage(parsed);
            }
          }
          if (closed) break;
        }
      }
      function rpc(method, params, timeoutMs) {
        var id = nextId++;
        var msg = { jsonrpc: '2.0', id: id, method: method, params: params === undefined ? {} : params };
        return new Promise(function (resolve, reject) {
          pending[id] = { resolve: resolve, reject: reject, timer: null };
          if (timeoutMs > 0) {
            pending[id].timer = setTimeout(function () {
              if (!pending[id]) return;
              var err = new Error('MCP call timed out after ' + timeoutMs + ' ms');
              settle(id, false, err);
            }, timeoutMs);
          }
          postMessage(msg).then(consumeResponse, function (e) { settle(id, false, e); })
            .then(function () {}, function (e2) { settle(id, false, e2); });
        });
      }
      var client = {
        transport: 'http',
        url: url,
        sessionId: function () { return sessionId; },
        protocolVersion: function () { return protocol; },
        set onNotification(fn) { noteCb = (typeof fn === 'function') ? fn : null; },
        get onNotification() { return noteCb; },
        call: function (method, params, opts) {
          if (closed) return Promise.reject(new Error('mcp: disconnected'));
          var timeout = 0;
          if (opts && typeof opts === 'object' && typeof opts.timeoutMs === 'number' && opts.timeoutMs > 0) {
            timeout = Math.min(600000, Math.floor(opts.timeoutMs));
          }
          var m = String(method), p = (params === undefined ? {} : params);
          if (m.indexOf('/') < 0) { p = { name: m, arguments: p || {} }; m = 'tools/call'; }
          return rpc(m, p, timeout);
        },
        listTools: function () { return rpc('tools/list', {}); },
        listResources: function () { return rpc('resources/list', {}); },
        disconnect: function () {
          if (closed) return;
          closed = true;
          var sids = Object.keys(pending);
          for (var i = 0; i < sids.length; i++) settle(Number(sids[i]), false, new Error('mcp: disconnected'));
          if (sessionId) {
            try { mcpFetch(url, { method: 'DELETE', headers: wireHeaders() }).catch(function () {}); } catch (e) {}
          }
        },
      };
      return (async function () {
        var init = await rpc('initialize', {
          protocolVersion: '2025-06-18', capabilities: {},
          clientInfo: { name: 'sofuu', version: '0.1.0' },
        }, 30000);
        if (init && typeof init.protocolVersion === 'string' && init.protocolVersion) protocol = init.protocolVersion;
        try {
          var ack = await postMessage({ jsonrpc: '2.0', method: 'notifications/initialized' });
          await consumeResponse(ack);
        } catch (eAck) {}
        return client;
      })();
    };

    sofuu.mcp.connect = function (arg, opts) {
      /* S6/M2: normalize both call shapes into the object the transports
       * understand. Config entries also carry {name, tools}; those are
       * caller-side fields and stay out of the child/HTTP spec. */
      var cfg = { command: '' };
      if (typeof arg === 'string') cfg.command = arg;
      else if (arg && typeof arg === 'object') {
        if (arg.command) cfg.command = String(arg.command);
        if (typeof arg.url === 'string') cfg.url = arg.url;
        if (arg.name) cfg.name = String(arg.name);
      }
      var src = (arg && typeof arg === 'object') ? arg : opts;
      if (src && typeof src === 'object') {
        if (Array.isArray(src.args)) cfg.args = src.args.map(String);
        if (src.env && typeof src.env === 'object') cfg.env = src.env;
        if (typeof src.cwd === 'string' && src.cwd) cfg.cwd = src.cwd;
        if (src.headers && typeof src.headers === 'object') cfg.headers = src.headers;
      }
      var transport = '';
      if (arg && typeof arg === 'object' && arg.transport) transport = String(arg.transport);
      else if (opts && typeof opts === 'object' && opts.transport) transport = String(opts.transport);
      if (transport === 'sse' || transport === 'http+sse') {
        return Promise.reject(new Error("mcp.connect: legacy HTTP+SSE transport is not implemented (use transport:'http', Streamable HTTP)"));
      }
      if (transport === 'http' && !cfg.url) {
        return Promise.reject(new Error("mcp.connect: transport 'http' needs a url"));
      }
      var useHttp = (transport === 'http') || (transport !== 'stdio' && !!cfg.url);
      var server = (opts && opts.server) ? String(opts.server)
        : (useHttp ? (cfg.name || cfg.url) : ((arg && typeof arg === 'object' && arg.server)
          ? String(arg.server) : cfg.command));
      /* S6/S4: the MCP allowlist is a policy decision on the server id (or
       * the raw command for a plain-script caller). */
      var pd = policyDecision('mcp.connect', { server: server, command: cfg.command || '' });
      if (pd && pd.decision === 'deny') {
        return Promise.reject(new Error('mcp.connect: denied by policy (' + pd.rule + '): ' +
          (pd.reason || '')));
      }
      try { cfg.env = resolveSecretEnv(cfg.env); }
      catch (eSec) { return Promise.reject(eSec); }
      function finishClient(client) {
        if (!client || client.__paged) return client;
        try {
          client.listTools = function () { return all(client, 'tools/list', 'tools'); };
          client.listResources = function () { return all(client, 'resources/list', 'resources'); };
          var defMs = policyCallTimeout();
          if (defMs > 0 && typeof client.call === 'function') {
            var realCall = client.call;
            client.call = function (method, params, copts) {
              var has = copts && typeof copts === 'object' && copts.timeoutMs !== undefined;
              if (!has) {
                var merged = { timeoutMs: defMs };
                if (copts && typeof copts === 'object') {
                  for (var ck in copts) {
                    if (Object.prototype.hasOwnProperty.call(copts, ck)) merged[ck] = copts[ck];
                  }
                }
                copts = merged;
              }
              return realCall.call(client, method, params, copts);
            };
            try { Object.defineProperty(client, '__callDefaulted', { value: true, enumerable: false }); }
            catch (eD) { client.__callDefaulted = true; }
          }
          try { Object.defineProperty(client, '__paged', { value: true, enumerable: false }); }
          catch (eP) { client.__paged = true; }
        } catch (eM) {}
        return client;
      }
      if (useHttp) {
        return Promise.resolve(sofuu.mcp._connectHttp(cfg)).then(finishClient, function (eHttp) {
          /* Transport 'auto' with a command: fall back to stdio. */
          if (transport === 'auto' && cfg.command) {
            var spec = { command: cfg.command, server: server };
            if (cfg.args) spec.args = cfg.args;
            if (cfg.env) spec.env = cfg.env;
            if (cfg.cwd) spec.cwd = cfg.cwd;
            return Promise.resolve(realConnect(spec)).then(finishClient);
          }
          throw eHttp;
        });
      }
      return Promise.resolve(realConnect(cfg)).then(finishClient);
    };
    sofuu.mcp.__paged = true;
  })();

  /* ── T2 (2026-10-23): one MCP tools/call → result mapping ────────
   * The old adapters read content[0].text only, which silently dropped
   * later blocks, images, resource links and structuredContent — and,
   * worse, turned an isError:true tool failure into a chatty success.
   * This is the single mapper both the agent and chat drivers use; the
   * returned shape matches the T1 tool contract. */
  sofuu.tools.mcpResult = function (res, opts) {
    opts = opts || {};
    var cap = (opts.maxChars | 0) || 12000;
    var blocks = (res && res.content) || [];
    var parts = [];
    var images = 0;
    for (var i = 0; i < blocks.length; i++) {
      var b = blocks[i] || {};
      var t = String(b.type || '');
      if (t === 'text') {
        parts.push(String(b.text == null ? '' : b.text));
      } else if (t === 'image' || t === 'audio') {
        var bytes = 0;
        try { bytes = Math.floor(String(b.data || '').length * 3 / 4); } catch (eB) {}
        parts.push('[' + t + ': ' + (b.mimeType || 'unknown') + ', ~' + bytes + ' bytes]');
        images++;
      } else if (t === 'resource') {
        var r = b.resource || {};
        var line = '[resource: ' + (r.uri || '') + (r.mimeType ? ' (' + r.mimeType + ')' : '') + ']';
        if (typeof r.text === 'string' && r.text) line += '\n' + r.text;
        parts.push(line);
      } else if (t === 'resource_link') {
        parts.push('[resource: ' + (b.uri || '') + (b.name ? ' (' + b.name + ')' : '') + ']');
      } else {
        try { parts.push(JSON.stringify(b)); } catch (eJ) { parts.push('[' + t + ']'); }
      }
    }
    var text = parts.join('\n');
    if (res && res.structuredContent && typeof res.structuredContent === 'object') {
      var sj = '';
      try { sj = JSON.stringify(res.structuredContent); } catch (eS) { sj = ''; }
      if (sj) text += (text ? '\n' : '') + '[structuredContent] ' + sj;
    }
    if (!text) text = '(empty tool result)';
    var truncated = false;
    if (cap > 0 && text.length > cap) {
      text = text.slice(0, cap) + '\n…[MCP result truncated at ' + cap + ' chars]';
      truncated = true;
    }
    var ok = !(res && res.isError === true);
    return {
      ok: ok,
      text: text,
      data: res ? res.structuredContent : undefined,
      error: ok ? undefined : { code: 'E_TOOL', message: 'MCP tool reported isError' },
      meta: { contentBlocks: blocks.length, images: images, truncated: truncated },
    };
  };

  /* S4: install the user/project policy when present. Missing files are a
   * no-op; a malformed one leaves the policy in place and is reported by an
   * explicit load() caller (we must not take a driver down over it). */
  try {
    if (sofuu.policy && typeof sofuu.policy.load === 'function') sofuu.policy.load();
  } catch (ePol) {}

  /* Domain toolpacks:
   *   - coding (alias code): file, shell, and patch tools
   *   - web: web search and reader
   *   - memory: semantic recall and durable learning
   *   - system: runtime and hardware diagnostics */
  var CODING_TOOLS = ['read_file', 'write_file', 'edit_file', 'apply_patch', 'grep', 'glob',
                      'list_dir', 'bash', 'code_eval', 'todo_write']
    .filter(function (n) { return !(SHELL_DISABLED && n === 'bash'); });

  sofuu.tools.PACKS = {
    coding: CODING_TOOLS,
    code: CODING_TOOLS,
    web: ['web_search', 'web_open'],
    memory: ['memory_search', 'memory_remember'],
    system: ['system_info'],
  };

  sofuu.tools.GROUP = CODING_TOOLS;
  sofuu.tools.ENABLED = { bash: !SHELL_DISABLED };

  /* ── T3 (2026-10-23): one lightweight schema validator ───────────
   * No external dependency. Covers type (incl. unions and integer),
   * required, properties (recursive), items (recursive) and enum.
   * `sofuu.tools.validateArgs` also backs the agent's responseSchema
   * check, so tool arguments and structured answers share one semantics. */
  function schemaTypeOk(v, t) {
    if (t === 'string') return typeof v === 'string';
    if (t === 'number') return typeof v === 'number' && isFinite(v);
    if (t === 'integer') return typeof v === 'number' && isFinite(v) && Math.floor(v) === v;
    if (t === 'boolean') return typeof v === 'boolean';
    if (t === 'array') return Array.isArray(v);
    if (t === 'object') return !!v && typeof v === 'object' && !Array.isArray(v);
    if (t === 'null') return v === null;
    return true;
  }
  function schemaWalk(v, schema, path) {
    if (!schema || typeof schema !== 'object') return null;
    if (Array.isArray(schema.enum) && schema.enum.length) {
      var found = false;
      for (var i = 0; i < schema.enum.length; i++) {
        if (schema.enum[i] === v) { found = true; break; }
      }
      if (!found) {
        return { path: path || '$', error: 'must be one of ' + JSON.stringify(schema.enum) };
      }
    }
    if (schema.type) {
      var types = Array.isArray(schema.type) ? schema.type : [schema.type];
      var okT = false;
      for (var ti = 0; ti < types.length; ti++) {
        if (schemaTypeOk(v, types[ti])) { okT = true; break; }
      }
      if (!okT) return { path: path || '$', error: 'must be ' + types.join('|') };
    }
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      var req = Array.isArray(schema.required) ? schema.required : [];
      for (var r = 0; r < req.length; r++) {
        if (!(req[r] in v)) {
          return { path: (path ? path + '.' : '') + req[r], error: 'is required' };
        }
      }
      var props = (schema.properties && typeof schema.properties === 'object')
        ? schema.properties : null;
      if (props) {
        for (var k in props) {
          if (!Object.prototype.hasOwnProperty.call(v, k) || v[k] === undefined) continue;
          var sub = schemaWalk(v[k], props[k], (path ? path + '.' : '') + k);
          if (sub) return sub;
        }
      }
    }
    if (Array.isArray(v) && schema.items) {
      var lim = Math.min(v.length, 200);
      for (var ai = 0; ai < lim; ai++) {
        var subI = schemaWalk(v[ai], schema.items, (path || '$') + '[' + ai + ']');
        if (subI) return subI;
      }
    }
    return null;
  }
  sofuu.tools.validateArgs = function (schema, value) {
    if (!schema || typeof schema !== 'object') return { ok: true };
    var bad = schemaWalk(value, schema, '$');
    return bad ? { ok: false, path: bad.path, error: bad.error } : { ok: true };
  };

  /* T3: tool inventory. Builtin (coding/system/memory) + web + host tools; MCP/inline tools are
   * per-run (they only exist after a def connects), so they are not in the
   * global list. `allowed`/`requiresApproval` are the ACTIVE policy's
   * verdicts, so a picker and the loop cannot disagree. Supports filtering
   * by category/pack via opts.category or opts.pack. */
  sofuu.tools.list = function (opts) {
    opts = opts || {};
    var enabledOnly = opts.enabledOnly !== false;
    var filterCat = opts.category || opts.pack || null;
    var out = [];
    function push(name, source, category, spec) {
      if (filterCat && filterCat !== category && filterCat !== source) return;
      if (enabledOnly && sofuu.tools.ENABLED && sofuu.tools.ENABLED[name] === false) return;
      /* H-01: the SAME request builder and evaluator the executor wrapper
       * uses (toolPolicyDecision → toolPolicyRequest), so an inventory row and
       * an execution cannot disagree. `conditional` marks a tool whose verdict
       * depends on call parameters (bash + shell.allow): it is not "denied",
       * it is "denied unless the command matches" — the per-call check in
       * executeWithPolicy is authoritative. */
      var d = toolPolicyDecision(name, null);
      var allowed = !(d && d.decision === 'deny');
      var requiresApproval = !!(d && d.decision === 'ask');
      out.push({
        name: name,
        source: source,
        category: category,
        description: String((spec && (spec.description || spec.summary)) || ''),
        schema: (spec && (spec.parameters || spec.schema)) || null,
        policy: {
          allowed: allowed,
          requiresApproval: requiresApproval,
          rule: (d && d.rule) || '',
          conditional: !!(d && d.conditional),
        },
      });
    }
    var i, keys, n;
    for (i = 0; i < sofuu.tools.GROUP.length; i++) {
      n = sofuu.tools.GROUP[i];
      if (sofuu.tools.TOOLS[n]) push(n, 'builtin', 'coding', sofuu.tools.TOOLS[n]);
    }
    if (sofuu.tools.TOOLS.system_info) {
      push('system_info', 'builtin', 'system', sofuu.tools.TOOLS.system_info);
    }
    if (sofuu.tools.TOOLS.memory_search) {
      push('memory_search', 'builtin', 'memory', sofuu.tools.TOOLS.memory_search);
    }
    if (sofuu.tools.TOOLS.memory_remember) {
      push('memory_remember', 'builtin', 'memory', sofuu.tools.TOOLS.memory_remember);
    }
    if (sofuu.web && sofuu.web.TOOLS) {
      keys = Object.keys(sofuu.web.TOOLS);
      for (i = 0; i < keys.length; i++) push(keys[i], 'web', 'web', sofuu.web.TOOLS[keys[i]]);
    }
    var host = globalThis.__sofuu_host_tools;
    if (host && typeof host === 'object') {
      keys = Object.keys(host);
      for (i = 0; i < keys.length; i++) push(keys[i], 'host', 'host', host[keys[i]]);
    }
    return out;
  };
  sofuu.tools.describe = function (name) {
    var wanted = String(name || '');
    var all = sofuu.tools.list({ enabledOnly: false });
    for (var i = 0; i < all.length; i++) {
      if (all[i].name === wanted) return all[i];
    }
    return null;
  };

  /* R2.3: kill every live tool subprocess (the cancel path calls this so a
   * cancelled run leaves no orphan child behind). Returns the kill count. */
  sofuu.tools.killActive = function () {
    var n = 0;
    while (LIVE_PROCS.length) {
      var h = LIVE_PROCS.pop();
      try { h.kill(); n++; } catch (eK) {}
    }
    return n;
  };

  /* ── H-01: install the one policy wrapper on EVERY executor ──────────
   * Runs last, after every TOOLS.* definition in this file (todo_write,
   * system_info and memory_* are added below the SOFUU_TOOLS section), so no
   * executor can escape it. Each spec keeps its body as __exec and exposes
   * execute = the wrapper. Idempotent: re-loading the driver (hot reload)
   * must not stack wrappers on wrappers.
   *
   * sofuu.tools.execute(name, params) is the documented public entry — the
   * remediation plan's validation uses it — and it goes through exactly the
   * same gate as TOOLS[name].execute. */
  (function installPolicyWrappers() {
    var names = Object.keys(TOOLS);
    for (var i = 0; i < names.length; i++) {
      var n = names[i];
      var spec = TOOLS[n];
      if (!spec || typeof spec.execute !== 'function' || spec.__policyWrapped) continue;
      spec.__exec = spec.execute;
      spec.__policyWrapped = true;
      spec.execute = (function (toolName) {
        return function (params, ctx) { return executeWithPolicy(toolName, params, ctx); };
      })(n);
    }
    sofuu.tools.execute = function (name, params, ctx) {
      return executeWithPolicy(name, params, ctx);
    };
  })();
})();
