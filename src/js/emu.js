/* src/js/emu.js — `sofuu.emu`: headless emulator sessions + MCP tools.
 *
 * PLAN-GPU-PLATFORM-UI-EMULATOR Phase E5. The reusable engine lives in the
 * standalone `sofuu-emu` companion binary; this driver is a thin, stateful
 * front end over `sofuu ui snapshot`/`sofuu ui emu` so agents can drive a
 * multi-step UI journey and get an audit report after every step.
 *
 * A session is deterministic replay: every action appends to a script, and a
 * capture re-runs the app with that script. That keeps the JS API stateless
 * across processes (no long-lived renderer to desynchronise) while actions,
 * keyboard/theme/orientation changes and hot-edited sources all take effect
 * on the next `screenshot()`.
 */
(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis : this;
  if (!g.sofuu) g.sofuu = {};
  var sofuu = g.sofuu;
  if (sofuu.emu && sofuu.emu._loaded) return;

  var emu = {};
  emu._loaded = true;

  var sessions = Object.create(null);
  var seq = 0;

  function nextId() {
    return 'emu-' + (++seq);
  }

  function exePath() {
    try {
      if (typeof process !== 'undefined' && process.argv && process.argv[0]) {
        return String(process.argv[0]);
      }
    } catch (e) { /* fall through */ }
    return 'sofuu';
  }

  function tmpPath(tag) {
    var dir = '/tmp';
    try {
      if (typeof process !== 'undefined' && process.env && process.env.TMPDIR) {
        dir = String(process.env.TMPDIR).replace(/[\\/]+$/, '');
      } else if (typeof sofuu.os !== 'undefined' && sofuu.os && typeof sofuu.os.tmpdir === 'function') {
        dir = String(sofuu.os.tmpdir());
      }
    } catch (e) { /* keep /tmp */ }
    return dir + '/sofuu-emu-' + tag + '-' + String(seq) + '-' +
      String(Math.floor(Math.random() * 1e6)) + '.png';
  }

  function lastJsonLine(text) {
    var lines = String(text || '').split('\n');
    for (var i = lines.length - 1; i >= 0; i--) {
      var line = lines[i].replace(/\r$/, '').trim();
      if (line && line.charAt(0) === '{') return line;
    }
    return '';
  }

  /** Parse a command's JSON payload: one compact line, or pretty-printed
   *  output (the profiles manifest) after any driver chatter. */
  function parsePayload(text) {
    var trimmed = String(text || '').trim();
    if (!trimmed) return null;
    var line = lastJsonLine(trimmed);
    if (line) {
      try { return JSON.parse(line); } catch (e) { /* try the whole body */ }
    }
    var start = trimmed.indexOf('{');
    if (start >= 0) {
      try { return JSON.parse(trimmed.slice(start)); } catch (e2) { /* no JSON */ }
    }
    return null;
  }

  /** Run the `sofuu` binary and resolve the JSON payload of the last line. */
  function runCli(args) {
    return new Promise(function (resolve, reject) {
      if (typeof sofuu.spawn !== 'function') {
        reject(new Error('sofuu.emu: sofuu.spawn is unavailable in this runtime'));
        return;
      }
      var out = '';
      var err = '';
      var child;
      try {
        child = sofuu.spawn({
          command: exePath(),
          args: args,
          onStdout: function (chunk) { out += String(chunk); },
          onStderr: function (chunk) { err += String(chunk); },
          onExit: function (code) {
            if (code !== 0) {
              var detail = err.trim() || out.trim();
              reject(new Error('sofuu ' + args.slice(0, 2).join(' ') +
                ' exited with ' + code + (detail ? ': ' + detail : '')));
              return;
            }
            var parsed = parsePayload(out);
            if (!parsed) {
              reject(new Error('sofuu.emu: command produced no JSON' +
                (err.trim() ? ': ' + err.trim() : '')));
              return;
            }
            resolve(parsed);
          },
        });
      } catch (e) {
        reject(e);
      }
      /* Keep the handle reachable for the lifetime of the capture. */
      void child;
    });
  }

  function envArgs(args, s) {
    args.push('--platform', String(s.env.platform));
    if (s.env.device) args.push('--device', String(s.env.device));
    args.push('--theme', String(s.env.theme));
    args.push('--orientation', String(s.env.orientation));
    if (s.env.keyboardOpen) args.push('--keyboard-open');
    if (s.env.scenario) args.push('--scenario', String(s.env.scenario));
    if (s.env.props !== null && s.env.props !== undefined) {
      args.push('--props', JSON.stringify(s.env.props));
    }
    if (s.env.scale) args.push('--scale', String(s.env.scale));
    if (s.env.mock) args.push('--mock');
    if (s.env.offline) args.push('--offline');
    if (s.env.mockConfig) args.push('--mock-config', JSON.stringify(s.env.mockConfig));
    if (s.env.slowmo) args.push('--slowmo', String(s.env.slowmo));
    if (s.actions.length) args.push('--actions', JSON.stringify(s.actions));
    return args;
  }

  /** `audit` view of a capture; the report is what the API promises. */
  function report(capture) {
    var audit = (capture && capture.audit) ? capture.audit : capture;
    if (!audit || typeof audit !== 'object') {
      throw new Error('sofuu.emu: capture returned no audit report');
    }
    if (capture) {
      /* Flatten capture metadata onto the report instead of nesting the
       * capture object: `capture.audit` points back at this report, and a
       * cyclic value would fail JSON.stringify in every MCP tool result. */
      var keys = ['ok', 'width', 'height', 'scale', 'mode', 'platform',
        'device', 'renderer', 'capture_ms', 'capture_png', 'audit_json'];
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        if (capture[k] !== undefined && audit[k] === undefined) audit[k] = capture[k];
      }
    }
    return audit;
  }

  function runCapture(s, opts) {
    opts = opts || {};
    s._assert();
    var args = ['ui', 'snapshot', s.app];
    envArgs(args, s);
    args.push('--mode', String(opts.mode || 'clean'));
    if (opts.out) args.push('--out', String(opts.out));
    if (opts.auditJson) args.push('--audit-json', String(opts.auditJson));
    args.push('--json');
    return runCli(args).then(report);
  }

  function Session(app, opts) {
    opts = opts || {};
    this.id = nextId();
    this.app = String(app || '');
    this.env = {
      platform: opts.platform || 'ios',
      device: opts.device || null,
      theme: opts.theme || 'light',
      orientation: opts.orientation || 'portrait',
      keyboardOpen: !!opts.keyboardOpen,
      scenario: opts.scenario || null,
      props: (opts.props !== undefined ? opts.props : null),
      scale: opts.scale || null,
      mock: !!opts.mock || !!opts.offline || !!opts.mockConfig,
      offline: !!opts.offline,
      mockConfig: opts.mockConfig || null,
      slowmo: opts.slowmo || null,
    };
    this.actions = [];
    this.closed = false;
    this.headless = opts.headless !== false;
    this.watch = !!opts.watch;
    this.process = null;
    this.workDir = opts.workDir || null;
  }

  Session.prototype._assert = function () {
    if (this.closed) throw new Error('sofuu.emu: session ' + this.id + ' is closed');
    if (!this.app) throw new Error('sofuu.emu: session has no app entry file');
  };

  /** Append one semantic action. Actions are replayed on the next capture. */
  Session.prototype.interact = function (action) {
    this._assert();
    if (Array.isArray(action)) {
      for (var i = 0; i < action.length; i++) this.interact(action[i]);
      return this;
    }
    if (!action || typeof action !== 'object') {
      throw new Error('sofuu.emu: interact() expects an action object or array');
    }
    var copy = {};
    for (var k in action) {
      if (Object.prototype.hasOwnProperty.call(action, k)) copy[k] = action[k];
    }
    if (!copy.type) copy.type = copy.id ? 'tap' : 'wait_settled';
    this.actions.push(copy);
    return this;
  };

  Session.prototype.tap = function (id) {
    return this.interact({ type: 'tap', id: id });
  };

  Session.prototype.type = function (id, text) {
    return this.interact({ type: 'type', id: id, text: String(text === undefined ? '' : text) });
  };

  Session.prototype.submit = function (id, text) {
    return this.interact({ type: 'submit', id: id, text: String(text === undefined ? '' : text) });
  };

  Session.prototype.toggle = function (id, checked) {
    return this.interact({ type: 'toggle', id: id, checked: checked !== false });
  };

  Session.prototype.selectSegment = function (id, index) {
    return this.interact({ type: 'segment', id: id, index: index || 0 });
  };

  Session.prototype.selectTab = function (id, index) {
    return this.interact({ type: 'tab', id: id, index: index || 0 });
  };

  Session.prototype.selectRow = function (id, rowId) {
    return this.interact({ type: 'select_row', id: id, row: rowId });
  };

  /** Replay a deterministic scroll gesture on a scroll container. */
  Session.prototype.scroll = function (id, deltaY) {
    return this.interact({ type: 'scroll', id: id, delta_y: Number(deltaY || 0) });
  };

  Session.prototype.waitSettled = function () {
    this._assert();
    return runCapture(this, { mode: 'clean', out: tmpPath('settle') }).then(function () {
      return this;
    }.bind(this));
  };

  Session.prototype.screenshot = function (path, opts) {
    if (path && typeof path === 'object' && opts === undefined) {
      opts = path;
      path = null;
    }
    opts = opts || {};
    return runCapture(this, {
      mode: opts.mode || this.mode || 'clean',
      out: path || opts.out || null,
      auditJson: opts.auditJson || null,
    });
  };

  Session.prototype.setKeyboard = function (open) {
    this._assert();
    this.env.keyboardOpen = !!open;
    return this;
  };

  Session.prototype.setTheme = function (theme) {
    this._assert();
    this.env.theme = String(theme || 'light');
    return this;
  };

  Session.prototype.setPlatform = function (platform) {
    this._assert();
    this.env.platform = String(platform || 'ios');
    return this;
  };

  Session.prototype.setDevice = function (device) {
    this._assert();
    this.env.device = device ? String(device) : null;
    return this;
  };

  Session.prototype.setOrientation = function (orientation) {
    this._assert();
    this.env.orientation = String(orientation || 'portrait');
    return this;
  };

  Session.prototype.rotate = function () {
    return this.setOrientation(this.env.orientation === 'landscape' ? 'portrait' : 'landscape');
  };

  Session.prototype.setScenario = function (scenario) {
    this._assert();
    this.env.scenario = scenario ? String(scenario) : null;
    return this;
  };

  Session.prototype.setProps = function (props) {
    this._assert();
    this.env.props = (props === undefined ? null : props);
    return this;
  };

  /** Change any environment field in one call; unknown keys are kept as props. */
  Session.prototype.setEnv = function (env) {
    this._assert();
    env = env || {};
    if (env.platform !== undefined) this.setPlatform(env.platform);
    if (env.device !== undefined) this.setDevice(env.device);
    if (env.theme !== undefined) this.setTheme(env.theme);
    if (env.orientation !== undefined) this.setOrientation(env.orientation);
    if (env.keyboardOpen !== undefined) this.setKeyboard(env.keyboardOpen);
    if (env.keyboard !== undefined) this.setKeyboard(env.keyboard);
    if (env.scenario !== undefined) this.setScenario(env.scenario);
    if (env.props !== undefined) this.setProps(env.props);
    if (env.scale !== undefined) this.env.scale = env.scale || null;
    if (env.mock !== undefined) this.env.mock = !!env.mock;
    if (env.offline !== undefined) {
      this.env.offline = !!env.offline;
      if (this.env.offline) this.env.mock = true;
    }
    if (env.mockConfig !== undefined) {
      this.env.mockConfig = env.mockConfig || null;
      if (this.env.mockConfig) this.env.mock = true;
    }
    if (env.slowmo !== undefined) this.env.slowmo = env.slowmo || null;
    return this;
  };

  Session.prototype.envInfo = function () {
    return {
      session: this.id,
      app: this.app,
      platform: this.env.platform,
      device: this.env.device,
      theme: this.env.theme,
      orientation: this.env.orientation,
      keyboardOpen: this.env.keyboardOpen,
      scenario: this.env.scenario,
      mock: this.env.mock,
      offline: this.env.offline,
      slowmo: this.env.slowmo,
      actions: this.actions.slice(),
      headless: this.headless,
    };
  };

  Session.prototype.close = function () {
    if (this.closed) return true;
    this.closed = true;
    this.actions.length = 0;
    delete sessions[this.id];
    if (this.process && typeof this.process.kill === 'function') {
      try { this.process.kill(); } catch (e) { /* already gone */ }
    }
    this.process = null;
    return true;
  };

  function spawnInteractive(s) {
    if (typeof sofuu.spawn !== 'function') return null;
    var args = ['ui', 'emu', s.app];
    envArgs(args, s);
    if (s.watch) args.push('--watch');
    try {
      return sofuu.spawn({
        command: exePath(),
        args: args,
        onStdout: function () {},
        onStderr: function () {},
        onExit: function () {},
      });
    } catch (e) {
      return null;
    }
  }

  function makeSession(app, opts) {
    var s = new Session(app, opts);
    sessions[s.id] = s;
    if (!s.headless) s.process = spawnInteractive(s);
    return s;
  }

  /**
   * Open a session.
   *
   *   var s = sofuu.emu.open('./app.js', { platform: 'android', headless: true });
   *   await s.tap('tab_settings').setKeyboard(true).screenshot('/tmp/shot.png', { mode: 'annotated' });
   */
  emu.open = function (app, opts) {
    if (!app || typeof app !== 'string') {
      throw new Error('sofuu.emu.open(app, opts): app entry path is required');
    }
    return makeSession(app, opts);
  };

  /** One-shot capture without keeping a session. */
  emu.screenshot = function (app, path, opts) {
    var s = makeSession(app, opts || {});
    var capture = s.screenshot(path, opts || {});
    return capture.then(function (audit) {
      s.close();
      return audit;
    }, function (e) {
      s.close();
      throw e;
    });
  };

  /** Platform/device manifest from `sofuu-emu profiles`. */
  emu.profiles = function () {
    return runCli(['ui', 'profiles']).then(function (manifest) {
      return manifest.profiles ? manifest : { profiles: manifest };
    });
  };

  function sessionFrom(id) {
    var s = sessions[String(id || '')];
    if (!s) throw new Error('ui_emu: unknown session `' + String(id || '') + '`');
    if (s.closed) throw new Error('ui_emu: session `' + id + '` is closed');
    return s;
  }

  var OPEN_SCHEMA = {
    type: 'object',
    properties: {
      app: { type: 'string', description: 'App entry file (JS) that mounts a sofuu.ui surface' },
      platform: { type: 'string', enum: ['ios', 'android', 'macos', 'windows', 'linux'] },
      device: { type: 'string', description: 'Device preset id (see ui_emu_profiles)' },
      theme: { type: 'string', enum: ['light', 'dark'] },
      orientation: { type: 'string', enum: ['portrait', 'landscape'] },
      keyboard: { type: 'boolean', description: 'Composite the software keyboard' },
      scenario: { type: 'string', description: 'Scenario name or alternate entry file' },
      props: { type: 'object', description: 'Preview props exposed as __sofuu_emu_props' },
      headless: { type: 'boolean', description: 'false launches the interactive window' },
      watch: { type: 'boolean', description: 'Interactive window hot-reloads on change' },
      scale: { type: 'number', description: 'Cap the physical pixel scale' },
      mock: { type: 'boolean', description: 'Stub fetch/sofuu.ai locally (no network)' },
      offline: { type: 'boolean', description: 'Reject unmocked fetches with E_EMU_OFFLINE' },
      mockConfig: { type: 'object', description: 'Mocked routes: {routes:[{url,method,status,json|text}]}' },
      slowmo: { type: 'number', description: 'Interactive pump pace (2 = half speed)' },
    },
    required: ['app'],
  };

  var INTERACT_SCHEMA = {
    type: 'object',
    properties: {
      session: { type: 'string' },
      type: { type: 'string', description: 'tap|type|submit|toggle|segment|tab|select_row|wait_settled' },
      id: { type: 'string' },
      text: { type: 'string' },
      checked: { type: 'boolean' },
      index: { type: 'number' },
      row: { type: 'string' },
      actions: {
        type: 'array',
        description: 'Array of action objects; each may use type/id/text/checked/index/row',
        items: { type: 'object' },
      },
      mode: { type: 'string', enum: ['clean', 'annotated'] },
      path: { type: 'string', description: 'PNG path (defaults to a temp file)' },
    },
    required: ['session'],
  };

  var SHOT_SCHEMA = {
    type: 'object',
    properties: {
      session: { type: 'string' },
      path: { type: 'string', description: 'PNG output path' },
      mode: { type: 'string', enum: ['clean', 'annotated', 'filmstrip'] },
      audit_json: { type: 'string', description: 'Also write the AuditReport JSON here' },
    },
    required: ['session'],
  };

  var ENV_SCHEMA = {
    type: 'object',
    properties: {
      session: { type: 'string' },
      platform: { type: 'string', enum: ['ios', 'android', 'macos', 'windows', 'linux'] },
      device: { type: 'string' },
      theme: { type: 'string', enum: ['light', 'dark'] },
      orientation: { type: 'string', enum: ['portrait', 'landscape'] },
      keyboard: { type: 'boolean' },
      scenario: { type: 'string' },
      props: { type: 'object' },
      scale: { type: 'number' },
      mock: { type: 'boolean' },
      offline: { type: 'boolean' },
      mockConfig: { type: 'object' },
      slowmo: { type: 'number' },
    },
    required: ['session'],
  };

  var CLOSE_SCHEMA = {
    type: 'object',
    properties: {
      session: { type: 'string' },
    },
    required: ['session'],
  };

  function reportPayload(capture) {
    /* Accept either a raw capture ({audit:{...}}) or an already-unwrapped
     * AuditReport ({summary:{...}}) — runCapture() returns the latter. */
    var audit = (capture && capture.audit) ? capture.audit : (capture || {});
    var summary = audit.summary || {};
    return {
      capture_png: (capture && capture.capture_png) || audit.capture_png,
      mode: (capture && capture.mode) || audit.mode,
      error_count: summary.errors || 0,
      warning_count: summary.warnings || 0,
      audit: audit,
    };
  }

  /**
   * Built-in MCP tool server (`sofuu ui mcp`).
   *
   * Tools: ui_emu_open, ui_emu_interact, ui_emu_screenshot, ui_emu_set_env,
   * ui_emu_close, ui_emu_profiles. Every open/interact/screenshot returns the
   * full AuditReport so an agent can gate on `audit.summary.errors`.
   */
  emu.mcp = function (opts) {
    if (!sofuu.mcp || typeof sofuu.mcp.serve !== 'function') {
      throw new Error('sofuu.emu.mcp: sofuu.mcp.serve is unavailable in this runtime');
    }
    opts = opts || {};
    var server = sofuu.mcp.serve({
      name: opts.name || 'sofuu-ui-emu',
      version: opts.version || '1.0.0',
    });
    var lastApp = opts.app || '';

    server.tool('ui_emu_open', {
      description: 'Open a persistent emulator session for an app and return its initial audit.',
      schema: OPEN_SCHEMA,
    }, function (args) {
      args = args || {};
      var app = args.app || lastApp;
      if (!app) throw new Error('ui_emu_open requires `app`');
      lastApp = app;
      var s = makeSession(app, args);
      return runCapture(s, { mode: 'clean', out: tmpPath('open') }).then(function (capture) {
        return {
          session: s.id,
          app: app,
          platform: capture.platform || s.env.platform,
          device: capture.device || s.env.device,
          env: s.envInfo(),
          initial: reportPayload(capture),
        };
      }, function (e) {
        s.close();
        throw e;
      });
    });

    server.tool('ui_emu_interact', {
      description: 'Apply tap/type/submit/toggle/segment/tab/select_row actions and return the resulting audit.',
      schema: INTERACT_SCHEMA,
    }, function (args) {
      args = args || {};
      var s = sessionFrom(args.session);
      var list = Array.isArray(args.actions) ? args.actions : null;
      if (list) {
        s.interact(list);
      } else {
        var action = {};
        for (var k in args) {
          if (Object.prototype.hasOwnProperty.call(args, k) &&
              k !== 'session' && k !== 'mode' && k !== 'path') {
            action[k] = args[k];
          }
        }
        s.interact(action);
      }
      var mode = args.mode === 'annotated' ? 'annotated' : 'clean';
      var out = args.path || tmpPath('interact');
      return runCapture(s, { mode: mode, out: out }).then(function (capture) {
        var payload = reportPayload(capture);
        payload.session = s.id;
        payload.platform = capture.platform || s.env.platform;
        payload.device = capture.device || s.env.device;
        payload.env = s.envInfo();
        return payload;
      });
    });

    server.tool('ui_emu_screenshot', {
      description: 'Capture clean/annotated/filmstrip PNG plus the AuditReport.',
      schema: SHOT_SCHEMA,
    }, function (args) {
      args = args || {};
      var s = sessionFrom(args.session);
      var mode = args.mode || 'annotated';
      return runCapture(s, {
        mode: mode,
        out: args.path || null,
        auditJson: args.audit_json || null,
      }).then(function (capture) {
        var payload = reportPayload(capture);
        payload.session = s.id;
        payload.platform = capture.platform || s.env.platform;
        payload.device = capture.device || s.env.device;
        payload.env = s.envInfo();
        return payload;
      });
    });

    server.tool('ui_emu_set_env', {
      description: 'Change platform/device/theme/orientation/keyboard/scenario for a session.',
      schema: ENV_SCHEMA,
    }, function (args) {
      args = args || {};
      var s = sessionFrom(args.session);
      s.setEnv(args);
      return {
        session: s.id,
        env: s.envInfo(),
        hint: 'Call ui_emu_screenshot (or ui_emu_interact) to render the new environment.',
      };
    });

    server.tool('ui_emu_close', {
      description: 'Close an emulator session and release its action script.',
      schema: CLOSE_SCHEMA,
    }, function (args) {
      args = args || {};
      var s = sessionFrom(args.session);
      s.close();
      return { session: args.session, closed: true };
    });

    server.tool('ui_emu_profiles', {
      description: 'List the five platform profiles, device presets, sizes and touch rules.',
      schema: { type: 'object', properties: {} },
    }, function () {
      return emu.profiles();
    });

    return server;
  };

  emu._sessions = function () {
    var out = [];
    for (var id in sessions) {
      if (Object.prototype.hasOwnProperty.call(sessions, id)) out.push(sessions[id].envInfo());
    }
    return out;
  };

  sofuu.emu = emu;
})();
